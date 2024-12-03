from rarr_freehal import RARRFreeHal
import argparse

def set_args():
    parser = argparse.ArgumentParser(description="Arguments for RARRFreeHal configuration")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input data")
    parser.add_argument("--dataset", type=str, required=True, help="nq(NQ), sqa(SQA), haluqa(HaluQA)")
    parser.add_argument("--model_path", type=str, default= "meta-llama/Llama-3.1-8B-Instruct", help="Name of the model to use")
    parser.add_argument("--search_url", type=str, default= "http://127.0.0.1:8000/", help="URL for search operations")
    # parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing (default: 1000)")
    return parser.parse_args()
    

def main():
    args = set_args()

    rarr_freehal = RARRFreeHal(args)
    # data, total_latency_avg = rarr_freehal.correct()
    data = rarr_freehal.correct()
    
    # print(f"average of total latency : {total_latency_avg}")
    

if __name__ == "__main__":
    main()
    
# python main.py --input_path ../data/nq/nq_data2.csv --dataset nq