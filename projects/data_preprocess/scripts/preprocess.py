import random
import copy
import json

from transformers import LlamaTokenizer

from projects.data_preprocess.utils import read_json, save_json


tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf', cache_dir='../../../packages')

meta_instruction = "You are an AI assistant whose name is Jupiter.\n- Jupiter is a dialogue psychological diagnosis model that is developed by YouKen. " \
                   "It is designed to have the greatly abilities of psychological counseling relationship building, problem assessment and conceptualization.\n\n"

summary_instruction = ['是否存在心理健康问题和需要关注的事项？', '我的心理健康是否存在问题，需要特别留意？', '我的心理健康是否有任何问题需要注意？',
                       '我是否有心理健康问题，需要关注的地方？', '是否存在需要注意的心理健康问题？', '我的心理健康是否存在任何问题，需要特别关注？',
                       '我是否需要注意心理健康问题和相关注意事项？', '是否有心理健康问题需要我留意和关注？', '我的心理健康是否有需要注意的地方？',
                       '是否存在我需要注意的心理健康问题？']

eos_token = '/s'


def preprocess_pd_generated():
    data = []

    with open('../raw_data/generated/dp_generated_zyg_7_03.txt', 'r', encoding='utf-8') as raw_file:
        raw_data = raw_file.readlines()
        raw_file.close()

    output_data, group = [], []
    dialogue = []
    for i in range(0, len(raw_data)):
        line = raw_data[i]

        if line[:2] == '医生':
            speaker = '医生'
            line = '<|Jupiter|>：' + line[3:]
        elif line[:2] == '患者':
            speaker = '患者'
            line = '<|Human|>：' + line[3:]
        elif line[:7] == '请根据前面对话':
            speaker = '患者'
            line = '<|Human|>：' + line
        elif line == '\n':
            speaker = None
            output_data.append(dialogue)
            dialogue = []
            continue

        if raw_data[i + 1][:2] == '医生':
            next_speaker = '医生'
        elif raw_data[i + 1][:2] == '患者':
            next_speaker = '患者'
        elif raw_data[i + 1][:7] == '请根据前面对话':
            next_speaker = '患者'
        elif raw_data[i + 1] == '\n':
            next_speaker = None
        else:
            next_speaker = speaker

        group.append(line)

        if next_speaker != speaker:
            dialogue.append(''.join(group).strip())
            group = []

    for idx, utterances in enumerate(output_data):
        template = {'conversation_id': idx + 1, 'meta_instruction': meta_instruction, 'num_turns': 0, 'chat': {}}
        chat_id = 0
        for i in range(0, len(utterances)):
            if '<|Jupiter|>：' in utterances[i]:
                chat = dict()
                chat_id += 1
                chat['Jupiter'] = utterances[i] + eos_token
                if i + 1 < len(utterances):
                    chat['Human'] = utterances[i + 1] + eos_token
                template['chat'][f'turn_{chat_id}'] = chat

        template['num_turns'] = chat_id

        data.append(template)

    save_json(data, '../output_data/pd_generated.json')


def preprocess_d4():
    all_data = []

    for file_name in ['train', 'val', 'test']:
        data_dir = f'../raw_data/D4/raw_data_{file_name}.json'
        raw_data = read_json(data_dir)
        output = []

        for idx, data in enumerate(raw_data):
            template = {'conversation_id': '', 'meta_instruction': meta_instruction, 'num_turns': 0, 'chat': {}}
            grouped_messages = []
            tmp = []
            last_speaker = None
            for dialog in data['log']:
                speaker = dialog['speaker']
                if speaker != last_speaker and len(tmp) > 0:
                    grouped_messages.append(tmp)
                    tmp = []
                last_speaker = speaker
                tmp.append(dialog)
            if len(tmp) > 0:
                grouped_messages.append(tmp)

            chat_id = 0
            chat = dict()
            for chat_idx, message in enumerate(grouped_messages):
                speaker = message[0]['speaker']
                if speaker == 'patient':
                    patient_str = '<|Human|>: ' + ' '.join([msg['text'] for msg in message])

                    if chat_idx == 0 and '好的' in patient_str:
                        patient_str = '医生你好'

                    if patient_str == '' or len(patient_str) == 0:
                        patient_str = '医生你好'

                    for str_ in ['刘医生', '张医生', '周医生', '吴医生', '朱医生']:
                        if str_ in patient_str:
                            patient_str = patient_str.replace(str_, '医生')

                else:
                    if chat_idx == 0:
                        continue
                    utterances = []
                    for msg in message:
                        action = msg['action'] if msg['action'] is not None else '其他'
                        if action is not None:
                            utterances.append(msg['text'])
                    action_str = '<|Action|>: ' + action
                    utt_str = '<|Jupiter|>: ' + ' '.join(utterances)

                    chat_id += 1
                    chat[f'turn_{chat_id}'] = {'Human': patient_str + eos_token, 'Action': action_str + eos_token,
                                               'Jupiter': utt_str + eos_token}

            chat[f'turn_{chat_id + 1}'] = {'Human': '<|Human|>: ' + summary_instruction[random.randint(0, 9)] + eos_token,
                                       'Action': '<|Action|>: Summary' + eos_token,
                                       'Jupiter': f'<|Jupiter|>: 根据本次问诊，目前可能存在的问题如下：{data["record"]["summary"]}'
                                                  f'请注意，以上只是一个初步的评估和建议。建议您寻求专业心理医生的帮助，以获取个性化的评估和更详细的计划。' + eos_token}

            template['num_turns'] = chat_id + 1
            template['conversation_id'] = f'{idx + 1}'
            output.append(template)
            template['chat'] = chat

        all_data.extend(output)

    save_json(all_data, f'../output_data/pd_d4.json')


def process_rogers():
    with open('../raw_data/book_case/rogers.json', encoding='GB18030') as raw_file:
        raw_data = json.load(raw_file)
    raw_file.close()
    output = []

    for idx, dialog in enumerate(raw_data):
        template = {'conversation_id': '', 'meta_instruction': meta_instruction, 'num_turns': 0, 'chat': {}}
        grouped_messages = []
        tmp = []
        last_speaker = None
        for utterance in dialog:
            speaker = utterance['role']
            if speaker != last_speaker and len(tmp) > 0:
                grouped_messages.append(tmp)
                tmp = []
            last_speaker = speaker
            tmp.append(utterance)
        if len(tmp) > 0:
            grouped_messages.append(tmp)

        chat_id = 0
        chat = dict()
        for chat_idx, message in enumerate(grouped_messages):
            speaker = message[0]['role']
            if speaker == '罗杰斯':
                doctor_str = '<|Jupiter|>: ' + ' '.join(msg['content'] for msg in message)
            else:
                if chat_idx == 0:
                    continue
                utterances = []
                for msg in message:
                    utterances.append(msg['content'])
                utt_str = '<|Human|>: ' + ' '.join(utterances)

                chat_id += 1
                chat[f'turn_{chat_id}'] = {'Jupiter': doctor_str + f'{eos_token}', 'Human': utt_str + f'{eos_token}'}

        template['num_turns'] = chat_id
        template['conversation_id'] = f'{idx + 1}'
        output.append(template)
        template['chat'] = chat

    save_json(output, '../output_data/pd_rogers.json')


def merge_all_pd_conversations():
    d4 = read_json('../output_data/pd_d4.json')
    generated = read_json('../output_data/pd_generated.json')
    rogers = read_json('../output_data/pd_rogers.json')

    all_data = d4 + generated + rogers
    random.shuffle(all_data)

    for idx, conversation in enumerate(all_data):
        conversation['conversation_id'] = idx + 1

    save_json(all_data, '../output_data/pd_merge.json')


def process_merge():
    raw_data = read_json('../output_data/pd_merge.json')

    new_sample = []
    for sample in raw_data:

        chat = sample['chat']
        num_turns = int(sample['num_turns'])

        instruction = sample['meta_instruction']
        instruction_ids = tokenizer(instruction)['input_ids']
        assert isinstance(instruction_ids, list) and len(instruction_ids) > 0

        input_ids = copy.deepcopy(instruction_ids)
        for i in range(num_turns):
            cur_turn_ids = []
            cur_turn = chat[f'turn_{i + 1}']
            for key, value in cur_turn.items():
                cur_ids = tokenizer.encode(value)
                assert isinstance(cur_ids, list) and len(cur_ids) > 0
                cur_turn_ids.extend(cur_ids)

            if len(input_ids + cur_turn_ids) > 2048:
                # 划动窗口
                input_ids.extend(cur_turn_ids)
                new_sample.append(tokenizer.decode(input_ids).replace(' ⁇  ', ''))
                input_ids = copy.deepcopy(instruction_ids)
                cur_turn_ids = []

            input_ids.extend(cur_turn_ids)

        if len(input_ids) == len(instruction_ids):
            continue

        assert 0 < len(input_ids) <= 2048
        new_sample.append(tokenizer.decode(input_ids).replace(' ⁇  ', ''))

    print(len(new_sample))
    save_json(new_sample, '../output_data/pd_split.json')


if __name__ == '__main__':
    process_merge()
