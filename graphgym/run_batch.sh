#!/usr/bin/env bash

CONFIG=${CONFIG:-maxcut_er}
GRID=${GRID:-er_hyperparams}
REPEAT=${REPEAT:-3}
MAX_JOBS=${MAX_JOBS:-8}
SLEEP=${SLEEP:-1}
MAIN=${MAIN:-main_graphgym}

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python configs_gen.py --config ../copt_graphgym/configs/${CONFIG}.yaml \
  --grid grids/${GRID}.txt \
  --out_dir ../copt_graphgym/configs
#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
bash parallel.sh ../copt_graphgym/configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP ../${MAIN}
# rerun missed / stopped experiments
bash parallel.sh ../copt_graphgym/configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP ../${MAIN}
# rerun missed / stopped experiments
bash parallel.sh ../copt_graphgym/configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP ../${MAIN}

# aggregate results for the batch
python agg_batch.py --dir ../results/${CONFIG}_grid_${GRID}
