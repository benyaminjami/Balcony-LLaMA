# Define the node numbers and the master node
NODES=(10 3)  # List of node numbers
MASTER_NODE=10  # Specify which node number is the master
SYNC_MASTER=false  # Flag to determine whether to sync and check on the master node
RUN_NAME="Smollm_1B_PT_Sorted_Meta"  # Specify a unique name for this run
NNODES=${#NODES[@]}
GRAD_ACCUMULATION=$((36 / NNODES))
CONFIG_FILE=/work/benyamin/smollm/nanotron/pre-training/smollm2/config_smollm2_1B_sorted_meta.yaml
BRANCH="sorted_general"
# Define user and node addresses
USER="benyamin"
NODES_ADDRESS=()
for NODE_NUMBER in "${NODES[@]}"; do
    if [ "$NODE_NUMBER" -lt 10 ]; then
        NODES_ADDRESS+=("lux-3-node-0$NODE_NUMBER")
    else
        NODES_ADDRESS+=("lux-3-node-$NODE_NUMBER")
    fi
done
