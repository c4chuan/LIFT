from src.prompts.prompts import *
import json

from src.prompts.prompts import GUIDANCE


def get_examples():
    results = []
    examples = EXAMPLES['LIFT']
    for example in examples:
        example_list = []
        example_list.append(example['query'])
        example_list.append(example['answer'])
        example_list.append(example['image'])
        results.append(example_list)
    return results


if __name__ == "__main__":
    ref_json_path = "/data/wangzhenchuan/Projects/LIFT/visualwebarena/src/prompts/vwa/jsons/p_som_cot_id_actree_3s_final.json"
    save_path = "/data/wangzhenchuan/Projects/LIFT/visualwebarena/src/prompts/vwa/jsons/lift.json"
    ref_json = json.load(open(ref_json_path))
    lift_json = {"intro": INTROS['LIFT'],
                 "agent_intro": INTROS['LIFT'],
                 "intro_w_reflections": INTROS['LIFT'],
                 "intro_wo_icl": INTROS['LIFT'],
                 "init_template": "URL: {url}\nOBJECTIVE: {objective}\nPREVIOUS ACTION: {previous_action}",
                 "template": "URL: {url}\nPREVIOUS ACTION: {previous_action}",
                 "meta_data": ref_json['meta_data'], 'examples': get_examples(), }
    json.dump(lift_json, open(save_path, 'w'), indent=4)

