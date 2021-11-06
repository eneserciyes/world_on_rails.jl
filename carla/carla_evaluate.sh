#!/bin/bash
export RESULTS=/home/enes/avg/WoR/world_on_rails.jl/results

export SCENARIOS=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json
export ROUTES=${LEADERBOARD_ROOT}/data/routes_devtest.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=1
export TEAM_AGENT=/home/enes/avg/WoR/world_on_rails.jl/carla/carla_agent.py
export PORT=2000
export TM_PORT=8035
export DEBUG=0
export CHECKPOINT_ENDPOINT=${RESULTS}/results.json
​
python $LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--track=SENSORS \
--scenarios=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json  \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--routes=${ROUTES} \
--checkpoint=${CHECKPOINT_ENDPOINT} 

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."