from cove_freehal import COVEFreeHal
import argparse

def set_args():
    parser = argparse.ArgumentParser(description="Arguments for Cove + FreeHal configuration")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input data")
    parser.add_argument("--dataset", type=str, required=True, help="nq(NQ), sqa(SQA), haluqa(HaluQA)")
    parser.add_argument("--model_path", type=str, default= "meta-llama/Llama-3.1-8B-Instruct", help="Name of the model to use")
    
    return parser.parse_args()
    

def main():
    args = set_args()

    cove_freehal = COVEFreeHal(args)
    data, total_latency_avg = cove_freehal.correct()
    
    print(f"average of total latency : {total_latency_avg}")
    

if __name__ == "__main__":
    main()
    
# python main.py --input_path ../data/nq/nq_data2.csv --dataset nq