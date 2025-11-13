import json
def main(path):
    data = json.load(open(path))
    pass

if __name__ == '__main__':
    main('/data/wangzhenchuan/Projects/LIFT/top_rewards_analysis/total_highest/rank1_step199_rec1.jsonl')