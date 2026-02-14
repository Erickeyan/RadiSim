import os, sys
import datetime
import dotenv
from pathlib import Path
from icecream import ic

dotenv.load_dotenv(override=True)


def main():
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    assert os.path.isdir(src_dir), f"error src/ - {src_dir}"
    ic(f"Working Dir: {src_dir}")
    os.chdir(src_dir)

    # region >>>torchrun
    command = (
        f"torchrun "
        f"--nproc_per_node=4 "
        f"--rdzv_endpoint=$HOSTE_NODE_ADDR "
        f"--master_port=29507 "
        f"-m training.main "
        f"--model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 "
        f"--train-data=TRUE "
        f"--data=your_data_csv "
        f"--warmup 2000 --lr 1e-4 --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 "
        f"--precision amp --workers 32  --grad-clip-norm 1.0 "
        f"--local-loss "
        f"--gather-with-grad "
        f"--batch-size 128 "
        f"--accum-freq 16 "
        f"--epochs 10 "
        f"--name {TRAIN_COMMAND__NAME} "
        f"--logs {TRAIN_COMMAND__LOGS} "
        f"--seed 0 "
        f"--report-to tensorboard "
        # f"2 > {TRAIN_COMMAND_ERR_TO}"
        "--siglip "
      )
    # endregion <<<

    # region >>> Duplicate train call
    # Save the command file(or sh/py) to train log folder
    Path(os.path.join(TRAIN_COMMAND__LOGS, TRAIN_COMMAND__NAME)).mkdir(exist_ok=True, parents=True)
    save_to = os.path.join(TRAIN_COMMAND__LOGS, TRAIN_COMMAND__NAME, "commmand_call_history.txt")
    with open(save_to, 'a') as f:
        f.write(f"{'=' * 40} {datetime.datetime.now()} {'=' * 40}\n")
        f.write(f"Train Session Desc: {Train_Session_Desc}\n\n")
        f.write(f"Command: \n")
        f.write(command.replace(" --", "\n--"))
        f.write("\n\n")

    # endregion <<< Duplicate train call

    os.system(command)
    # subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    SCI_PROJECT_NFS_ROOT = dotenv.dotenv_values()["SCI_PROJECT_NFS_ROOT"]
    ic(SCI_PROJECT_NFS_ROOT)

    N_Nodes = 1
    NPROC_PER_NODE = 4 
    TRAIN_COMMAND__LOGS = "./train_log/"
    TRAIN_COMMAND__NAME = f"{N_Nodes}_{NPROC_PER_NODE}"
    TRAIN_COMMAND_ERR_TO = os.path.join(TRAIN_COMMAND__LOGS, TRAIN_COMMAND__NAME, "sci_err.log")

    Train_Session_Desc = ""

    main()