/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2015, Rutgers the State University of New Jersey, New Brunswick
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Rutgers University nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Authors: Zakary Littlefield */

#include "../HySST.h"
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/base/objectives/MinimaxObjective.h"
#include "ompl/base/objectives/MaximizeMinClearanceObjective.h"
#include "ompl/base/objectives/PathLengthOptimizationObjective.h"
#include "ompl/tools/config/SelfConfig.h"
#include "ompl/base/spaces/RealVectorStateSpace.h"
#include <limits>
#include <ompl/base/goals/GoalState.h>

ompl::geometric::HySST::HySST(const base::SpaceInformationPtr &si) : base::Planner(si, "HySST")
{
    specs_.approximateSolutions = true;
    specs_.directed = true;
    prevSolution_.clear();

    Planner::declareParam<double>("range", this, &HySST::setRange, &HySST::getRange, ".1:.1:100");
    Planner::declareParam<double>("goal_bias", this, &HySST::setGoalBias, &HySST::getGoalBias, "0.:.05:1.");
    Planner::declareParam<double>("selection_radius", this, &HySST::setSelectionRadius, &HySST::getSelectionRadius, "0.:.1:"
                                                                                                                    "100");
    Planner::declareParam<double>("pruning_radius", this, &HySST::setPruningRadius, &HySST::getPruningRadius, "0.:.1:100");

    addPlannerProgressProperty("best cost REAL", [this]
                               { return std::to_string(this->prevSolutionCost_.value()); });
}

ompl::geometric::HySST::~HySST()
{
    freeMemory();
}

void ompl::geometric::HySST::setup()
{
    base::Planner::setup();
    if (!nn_)
        nn_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    nn_->setDistanceFunction([this](const Motion *a, const Motion *b)
                             { return distanceFunction(a, b); });
    if (!witnesses_)
        witnesses_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    witnesses_->setDistanceFunction([this](const Motion *a, const Motion *b)
                                    { return distanceFunction(a, b); });

    if (pdef_)
    {
        if (pdef_->hasOptimizationObjective())
        {
            opt_ = pdef_->getOptimizationObjective();
            if (dynamic_cast<base::MaximizeMinClearanceObjective *>(opt_.get()) ||
                dynamic_cast<base::MinimaxObjective *>(opt_.get()))
                OMPL_WARN("%s: Asymptotic near-optimality has only been proven with Lipschitz continuous cost "
                          "functions w.r.t. state and control. This optimization objective will result in undefined "
                          "behavior",
                          getName().c_str());
        }
        else
        {
            OMPL_WARN("%s: No optimization object set. Using path length", getName().c_str());
            opt_ = std::make_shared<base::PathLengthOptimizationObjective>(si_);
            pdef_->setOptimizationObjective(opt_);
        }
    }
    else
    {
        OMPL_WARN("%s: No optimization object set. Using path length", getName().c_str());
        opt_ = std::make_shared<base::PathLengthOptimizationObjective>(si_);
    }
    prevSolutionCost_ = opt_->infiniteCost();
}

void ompl::geometric::HySST::clear()
{
    Planner::clear();
    sampler_.reset();
    freeMemory();
    if (nn_)
        nn_->clear();
    if (witnesses_)
        witnesses_->clear();
    if (opt_)
        prevSolutionCost_ = opt_->infiniteCost();
}

void ompl::geometric::HySST::freeMemory()
{
    if (nn_)
    {
        std::vector<Motion *> motions;
        nn_->list(motions);
        for (auto &motion : motions)
        {
            if (motion->state)
                si_->freeState(motion->state);
            delete motion;
        }
    }
    if (witnesses_)
    {
        std::vector<Motion *> witnesses;
        witnesses_->list(witnesses);
        for (auto &witness : witnesses)
        {
            if (witness->state)
                si_->freeState(witness->state);
            delete witness;
        }
    }

    for (auto &i : prevSolution_)
    {
        if (i)
            si_->freeState(i);
    }
    prevSolution_.clear();
}

ompl::geometric::HySST::Motion *ompl::geometric::HySST::selectNode(ompl::geometric::HySST::Motion *sample)
{
    std::vector<Motion *> ret; // List of all nodes within the selection radius
    Motion *selected = nullptr;
    base::Cost bestCost = opt_->infiniteCost();
    nn_->nearestR(sample, selectionRadius_, ret); // Find the nearest nodes within the selection radius of the random sample
    for (auto &i : ret)                           // Find the active node with the best cost within the selection radius
    {
        if (!i->inactive_ && opt_->isCostBetterThan(i->accCost_, bestCost))
        {
            bestCost = i->accCost_;
            selected = i;
        }
    }
    if (selected == nullptr) // However, if there are no active nodes within the selection radius, select the next nearest node
    {
        int k = 1;
        while (selected == nullptr)
        {
            nn_->nearestK(sample, k, ret);                                       // sample the k nearest nodes to the random sample into ret
            for (unsigned int i = 0; i < ret.size() && selected == nullptr; i++) // Find the active node with the best cost
                if (!ret[i]->inactive_)
                    selected = ret[i];
            k += 5; // If none found, increase the number of nearest nodes to sample FIXME: Increasingly expensive
        }
    }
    return selected;
}

ompl::geometric::HySST::Witness *ompl::geometric::HySST::findClosestWitness(ompl::geometric::HySST::Motion *node)
{
    if (witnesses_->size() > 0)
    {
        auto *closest = static_cast<Witness *>(witnesses_->nearest(node));
        if (distanceFunction(closest, node) > pruningRadius_) // If the closest witness is outside the pruning radius, return a new witness at the same point as the node.
        {
            closest = new Witness(si_);
            closest->linkRep(node);
            si_->copyState(closest->state, node->state);
            witnesses_->add(closest);
        }
        return closest;
    }
    else
    {
        auto *closest = new Witness(si_);
        closest->linkRep(node);
        si_->copyState(closest->state, node->state);
        witnesses_->add(closest);
        return closest;
    }
}

std::vector<ompl::geometric::HySST::Motion *> ompl::geometric::HySST::extend(Motion *parentMotion, base::Goal *goalState)
{
    // Vectors for storing flow and jump inputs
    std::vector<double> flowInputs;
    std::vector<double> jumpInputs;

    const unsigned int TF_INDEX = si_->getStateDimension() - 2; // Second to last column
    const unsigned int TJ_INDEX = si_->getStateDimension() - 1; // Last column

    auto *newMotion = new Motion(si_);
    newMotion->parent = parentMotion;

    // Generate random maximum flow time
    double random = rand();
    double randomFlowTimeMax = random / RAND_MAX * tM_;
    double tFlow = 0; // Tracking variable for the amount of flow time used in a given continuous simulation step

    bool collision = false; // Set collision to false initially

    // Choose whether to begin growing the tree in the flow or jump regime
    bool in_jump = jumpSet_(parentMotion->state);
    bool in_flow = flowSet_(parentMotion->state);
    bool priority = in_jump && in_flow ? random / RAND_MAX > 0.5 : in_jump; // If both are true, equal chance of being in flow or jump set.

    // Sample and instantiate parent vertices and states in edges
    base::State *previousState = si_->allocState();
    si_->copyState(previousState, parentMotion->state);
    auto *collisionParentMotion = parentMotion; // used to point to nn_->nearest(randomMotion);

    // Allocate memory for the new edge
    std::vector<base::State *> *intermediateStates = new std::vector<base::State *>;

    // Fill the edge with the starting vertex
    base::State *parentState = si_->allocState();
    si_->copyState(parentState, previousState);
    intermediateStates->push_back(parentState);

    // Simulate in either the jump or flow regime
    if (!priority)
    { // Flow
        // Randomly sample the flow inputs
        flowInputs = sampleFlowInputs_();

        while (tFlow < randomFlowTimeMax && flowSet_(parentMotion->state))
        {
            tFlow += flowStepDuration_;

            // Find new state with continuous simulation
            base::State *intermediateState = si_->allocState();
            intermediateState = this->continuousSimulator_(flowInputs, previousState, flowStepDuration_, intermediateState);
            intermediateState->as<base::RealVectorStateSpace::StateType>()->values[TF_INDEX] += flowStepDuration_;

            // Return nullptr if in unsafe set and exit function
            if (unsafeSet_(intermediateState))
                return std::vector<Motion *>(); // Return empty vector

            // Add new intermediate state to edge
            intermediateStates->push_back(intermediateState);

            // Collision Checking
            double ts = parentMotion->state->as<base::RealVectorStateSpace::StateType>()->values[TF_INDEX];
            double tf = intermediateState->as<base::RealVectorStateSpace::StateType>()->values[TF_INDEX];

            collision = collisionChecker_(intermediateStates, jumpSet_, ts, tf, intermediateState, TF_INDEX);

            // State has passed all tests so update parent, edge, and temporary states
            si_->copyState(newMotion->state, intermediateState); // Set parent state for next iterations
            si_->copyState(previousState, intermediateState);

            // Add motion to tree or handle collision/goal
            bool inGoalSet = goalState->isSatisfied(intermediateState, &tolerance_);
            // bool inGoalSet = distanceFunc_(intermediateState, goalState->as<base::GoalState>()->getState()) <= tolerance_;    // TODO: Substitute with goal->isSatisfied()

            // If maximum flow time has been reached, a collision has occured, or a solution has been found, exit the loop
            if (tFlow >= randomFlowTimeMax || collision || inGoalSet)
            {
                // Create motion to add to tree
                auto *motion = new Motion(si_);
                si_->copyState(motion->state, intermediateState);
                motion->parent = parentMotion;
                motion->edge = intermediateStates; // Set the new motion edge

                if (inGoalSet)
                {
                    // newMotion->edge = intermediateStates;
                    newMotion->edge = intermediateStates;
                    std::cout << "possible solution found!" << std::endl;
                    return std::vector<Motion *>{motion};
                }
                else if (collision)
                {
                    collisionParentMotion = motion;
                    priority = true; // If collision has occurred, continue to jump regime
                }
                else
                    return std::vector<Motion *>{motion}; // Return the motion in vector form
                // nn_->add(motion);
                break;
            }
        }
    }

    if (priority)
    { // Jump
        // Randomly sample the jump inputs
        jumpInputs = sampleJumpInputs_();

        // Instantiate and find new state with discrete simulator
        base::State *newState = si_->allocState();

        // OMPL_INFORM("  pre-jump x-vel: %f", newMotion->state->as<base::RealVectorStateSpace::StateType>()->values[2]);
        // OMPL_INFORM("  pre-jump y-vel: %f", newMotion->state->as<base::RealVectorStateSpace::StateType>()->values[3]);

        newState = this->discreteSimulator_(newMotion->state, jumpInputs, newState); // changed from previousState to newMotion->state
        newState->as<base::RealVectorStateSpace::StateType>()->values[TJ_INDEX]++;

        // OMPL_INFORM("  post-jump x-vel: %f", newState->as<base::RealVectorStateSpace::StateType>()->values[2]);
        // OMPL_INFORM("  post-jump y-vel: %f", newState->as<base::RealVectorStateSpace::StateType>()->values[3]);

        // Return nullptr if in unsafe set and exit function
        if (unsafeSet_(newState))
        {
            // OMPL_INFORM("  in unsafe set");
            return std::vector<Motion *>(); // Return empty vector
        }

        si_->copyState(newMotion->state, newState); // Set parent for next iteration

        // Create motion to add to tree
        auto *motion = new Motion(si_);
        si_->copyState(motion->state, newState);
        motion->parent = collisionParentMotion;

        // Add motions to tree, and free up memory allocated to newState
        return std::vector<Motion *>{motion, collisionParentMotion};

        // nn_->add(motion);
        // nn_->add(collisionParentMotion);
        si_->freeState(newState);
    }
}

void ompl::geometric::HySST::randomSample(Motion *randomMotion)
{
    sampler_ = si_->allocStateSampler();
    // Replace later with the ompl sampler, for now leave as custom
    sampler_->sampleUniform(randomMotion->state);
}

ompl::base::PlannerStatus ompl::geometric::HySST::solve(const base::PlannerTerminationCondition &ptc)
{
    checkValidity();
    checkMandatoryParametersSet();
    base::Goal *goal = pdef_->getGoal().get();
    auto *goal_s = dynamic_cast<base::GoalSampleableRegion *>(goal);

    if (goal_s == nullptr)
    {
        OMPL_ERROR("%s: Unknown type of goal", getName().c_str());
        return base::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
    }

    if (!goal_s->couldSample())
    {
        OMPL_ERROR("%s: Insufficient states in sampleable goal region", getName().c_str());
        return base::PlannerStatus::INVALID_GOAL;
    }

    while (const base::State *st = pis_.nextStart())
    {
        auto *motion = new Motion(si_);
        si_->copyState(motion->state, st);
        nn_->add(motion);
        motion->accCost_ = opt_->identityCost(); // Initialize the accumulated cost to the identity cost
        findClosestWitness(motion);              // Set representatives for the witness set
    }

    if (nn_->size() == 0)
    {
        OMPL_ERROR("%s: There are no valid initial states!", getName().c_str());
        return base::PlannerStatus::INVALID_START;
    }

    // if (!sampler_)
    // sampler_ = si_->allocStateSampler();

    const base::ReportIntermediateSolutionFn intermediateSolutionCallback = pdef_->getIntermediateSolutionCallback();

    OMPL_INFORM("%s: Starting planning with %u states already in datastructure", getName().c_str(), nn_->size());

    Motion *solution = nullptr;
    Motion *approxsol = nullptr;
    double approxdif = std::numeric_limits<double>::infinity();
    bool sufficientlyShort = false;
    auto *rmotion = new Motion(si_);
    base::State *rstate = rmotion->state;
    base::State *xstate = si_->allocState();

    unsigned iterations = 0;

    while (ptc == false)
    {
        /* sample random state (with goal biasing) */
        bool attemptToReachGoal = (rng_.uniform01() < goalBias_ && goal_s->canSample());
        if (attemptToReachGoal)
            goal_s->sampleGoal(rstate);
        else
            randomSample(rmotion);
        // sampler_->sampleUniform(rstate);

        /* find closest state in the tree */
        Motion *nmotion = selectNode(rmotion);

        std::vector<Motion *> dMotion = {new Motion(si_)};
        double d = si_->distance(nmotion->state, rstate);

        // attemptToReachGoal = rng_.uniform01() < .5;
        // if (attemptToReachGoal)
        // {
        //     if (d > maxDistance_)
        //     {
        //         si_->getStateSpace()->interpolate(nmotion->state, rstate, maxDistance_ / d, xstate);
        //         dMotion[0]->state = xstate;
        //     }
        // }
        // else
        // {
        checkMandatoryParametersSet();
        dMotion = extend(nmotion, goal);
        // }

        if (dMotion.size() == 0) // If extension failed, continue to next iteration
        {
            // si_->freeState(rstate);
            continue;
        }

        si_->copyState(rstate, dMotion[0]->state); // copy the new state to the random state pointer first value will always be the newest state, even if a collision occurs

        // if (si_->checkMotion(nmotion->state, rstate)) // If motion is valid -- we know it is valid because the extend function checks for collisions + unsafe set
        // {
        base::Cost incCost = opt_->motionCost(nmotion->state, rstate);    // Compute incremental cost
        base::Cost cost = opt_->combineCosts(nmotion->accCost_, incCost); // Combine total cost
        Witness *closestWitness = findClosestWitness(rmotion);            // Find closest witness

        // // Print the results of the two conditions below
        // std::cout << "Is the newly propagated state a child of the new representative of the witness? " << (closestWitness->rep_ == rmotion) << std::endl;
        // std::cout << "Does the newly propagated state dominate the old representative's cost? " << opt_->isCostBetterThan(cost, closestWitness->rep_->accCost_) << std::endl;

        if (closestWitness->rep_ == rmotion || opt_->isCostBetterThan(cost, closestWitness->rep_->accCost_)) // If the newly propagated state is a child of the new representative of the witness (previously had no rep) or it dominates the old representative's cost
        {
            // Print the results of the two conditions below
            // std::cout << "Is the newly propagated state a child of the new representative of the witness? " << (closestWitness->rep_ == rmotion) << std::endl;
            // std::cout << "Does the newly propagated state dominate the old representative's cost? " << opt_->isCostBetterThan(cost, closestWitness->rep_->accCost_) << std::endl;
            //     OMPL_INFORM("  x-value: %f", rstate->as<base::RealVectorStateSpace::StateType>()->values[0]);
            //     OMPL_INFORM("  y-value: %f", rstate->as<base::RealVectorStateSpace::StateType>()->values[1]);
            //     OMPL_INFORM("  t-value: %f", rstate->as<base::RealVectorStateSpace::StateType>()->values[8]);
            //     OMPL_INFORM("  j-value: %f", rstate->as<base::RealVectorStateSpace::StateType>()->values[9]);
            std::cout << rstate->as<base::RealVectorStateSpace::StateType>()->values[0] << " " << rstate->as<base::RealVectorStateSpace::StateType>()->values[1] << " " << rstate->as<base::RealVectorStateSpace::StateType>()->values[8] << " " << rstate->as<base::RealVectorStateSpace::StateType>()->values[9] << std::endl;

            Motion *oldRep = closestWitness->rep_; // Set a copy oof the old representative
            /* create a motion copy  of the newly propagated state */
            auto *motion = new Motion(si_);
            auto *collisionParentMotion = new Motion(si_);
            motion->accCost_ = cost;
            si_->copyState(motion->state, rstate);

            if (dMotion.size() > 1)
            {
                collisionParentMotion->accCost_ = cost;
                si_->copyState(collisionParentMotion->state, dMotion[1]->state);
            }

            // if (!attemptToReachGoal)    // If not trying to reach goal, free dstate TODO: review what the purpose of this is
            //     for(auto &i : dMotion)
            //         si_->freeState(i->state);
            motion->parent = nmotion;
            nmotion->numChildren_++;
            closestWitness->linkRep(motion); // Create new edge and set the new node as the representative TODO: Do we not see if collision state can be linked?

            // std::cout << "added motion to representative set" << std::endl;

            nn_->add(motion); // Add new node to tree
            if (dMotion.size() > 1)
                nn_->add(collisionParentMotion);

            double dist = 0.0;
            // bool solv = goal->isSatisfied(motion->state, &dist);
            dist = distanceFunc_(motion->state, pdef_->getGoal()->as<base::GoalState>()->getState());
            bool solv = dist <= 0.6;
            if (solv && opt_->isCostBetterThan(motion->accCost_, prevSolutionCost_)) // If the new state is a solution and it has a lower cost than the previous solution
            {
                approxdif = dist;
                solution = motion;

                for (auto &i : prevSolution_) // Free the previous solution
                    if (i)
                        si_->freeState(i);
                prevSolution_.clear();
                Motion *solTrav = solution; // Traverse the solution and save the states in prevSolution_
                while (solTrav != nullptr)
                {
                    prevSolution_.push_back(si_->cloneState(solTrav->state));
                    solTrav = solTrav->parent;
                }
                prevSolutionCost_ = solution->accCost_;

                OMPL_INFORM("Found solution with cost %.2f, a distance %.2f away from goal", solution->accCost_.value(), dist);
                if (intermediateSolutionCallback) // If a callback is defined, call it TODO: What is intermediateSolution
                {
                    // the callback requires a vector with const elements -> create a copy
                    std::vector<const base::State *> prevSolutionConst(prevSolution_.begin(), prevSolution_.end());
                    intermediateSolutionCallback(this, prevSolutionConst, prevSolutionCost_);
                }
                sufficientlyShort = opt_->isSatisfied(solution->accCost_); // If the solution is sufficiently short (according to optimization ojective), stop search TODO: Check what happens if no optimization objective
                if (sufficientlyShort)
                {
                    break;
                }
            }
            if (solution == nullptr && dist < approxdif) // If no solution found and distance to goal of this new state is closer than before (because no gurantee of probabilistic completeness)
            {
                approxdif = dist;
                approxsol = motion;

                for (auto &i : prevSolution_)
                {
                    if (i)
                        si_->freeState(i);
                }
                prevSolution_.clear();
                Motion *solTrav = approxsol;
                while (solTrav != nullptr)
                {
                    prevSolution_.push_back(si_->cloneState(solTrav->state));
                    solTrav = solTrav->parent;
                }
            }

            if (oldRep != rmotion) // If the representative has changed (prune)
            {
                while (oldRep->inactive_ && oldRep->numChildren_ == 0) // While the current node is inactive and is a leaf
                {
                    oldRep->inactive_ = true; // Mark the node as inactive (redundant??)
                    nn_->remove(oldRep);      // Remove from list of active nodes

                    if (oldRep->state)
                        si_->freeState(oldRep->state);

                    oldRep->state = nullptr;
                    oldRep->parent->numChildren_--;
                    Motion *oldRepParent = oldRep->parent;
                    delete oldRep;
                    oldRep = oldRepParent;
                }
            }
        }
        // }
        iterations++;
    }

    bool solved = false;
    bool approximate = false;
    if (solution == nullptr)
    {
        solution = approxsol;
        approximate = true;
    }

    if (solution != nullptr)
    {
        /* set the solution path */
        auto path(std::make_shared<PathGeometric>(si_));
        for (int i = prevSolution_.size() - 1; i >= 0; --i)
            path->append(prevSolution_[i]);
        solved = true;
        pdef_->addSolutionPath(path, approximate, approxdif, getName());
    }

    si_->freeState(xstate);
    if (rmotion->state)
        si_->freeState(rmotion->state);
    rmotion->state = nullptr;
    delete rmotion;

    OMPL_INFORM("%s: Created %u states in %u iterations", getName().c_str(), nn_->size(), iterations);

    return {solved, approximate};
}

void ompl::geometric::HySST::getPlannerData(base::PlannerData &data) const
{
    Planner::getPlannerData(data);

    std::vector<Motion *> motions;
    std::vector<Motion *> allMotions;
    if (nn_)
        nn_->list(motions);

    for (auto &motion : motions)
        if (motion->numChildren_ == 0)
            allMotions.push_back(motion);
    for (unsigned i = 0; i < allMotions.size(); i++)
        if (allMotions[i]->getParent() != nullptr)
            allMotions.push_back(allMotions[i]->getParent());

    if (prevSolution_.size() != 0)
        data.addGoalVertex(base::PlannerDataVertex(prevSolution_[0]));

    for (auto &allMotion : allMotions)
    {
        if (allMotion->getParent() == nullptr)
            data.addStartVertex(base::PlannerDataVertex(allMotion->getState()));
        else
            data.addEdge(base::PlannerDataVertex(allMotion->getParent()->getState()),
                         base::PlannerDataVertex(allMotion->getState()));
    }
}