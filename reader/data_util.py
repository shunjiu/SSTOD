import copy


def read_name_dialog_from_file(data_json):
    data = {}
    for name, dialog in data_json.items():
        data_iter = {'goal': dialog['goal_value']['name'], 'log': [], 'results': dialog['results']}
        turn = 0
        turn_num = 0
        assert 'staff' in dialog['turns'][0]
        if 'staff' not in dialog['turns'][-1]:
            dialog['turns'].append({
                'staff': '好的，再见',
                'staff_label': [{
                    'action': "bye",
                    "param-knowledge": [],
                    "param-name": None
                }],
                'staff_state': dialog['turns'][-1]['user_state']
            })

        turn += 1
        data_turn_iter = {}
        while turn < len(dialog['turns']):
            data_turn_iter['user'] = dialog['turns'][turn]['user']
            user_label = dialog['turns'][turn]['user_label'][0]
            # data_turn_iter['user_act'] = user_label['action']

            kd_span = []
            asr_char = {}
            for kd in user_label['param-knowledge']:
                correct_char = kd['correct_char']
                char_knowledge = [t['string'] for t in kd['knowledge']]
                kd_span.append((correct_char, ' '.join(char_knowledge)))

            data_turn_iter['asr_char'] = dict(asr_char)
            data_turn_iter['kdpn'] = ''
            correct_char_list = []
            for kd in kd_span:
                if len(kd[1]) > 0:
                    data_turn_iter['kdpn'] += ' <kd> ' + kd[1]
                    correct_char_list.append(kd[0])
                else:
                    data_turn_iter['kdpn'] += ''
            data_turn_iter['correct_char'] = ','.join(correct_char_list)

            turn += 1
            data_turn_iter['sys'] = dialog['turns'][turn]['staff']
            staff_label = dialog['turns'][turn]['staff_label'][0]
            sys_action = staff_label['action']
            data_turn_iter['bs'] = ','.join([sample['char'] for sample in dialog['turns'][turn]['staff_state']])

            db = []
            action_param = ''
            for knowledge in staff_label['param-knowledge']:
                char = knowledge['char']
                action_param += char
                kd = ' '.join([kd['string'] for kd in knowledge['knowledge']])
                if len(kd) > 0:
                    db.append(char + ':' + kd)
            data_turn_iter['sys_act'] = sys_action
            data_turn_iter['sys_act_param'] = action_param
            data_turn_iter['db'] = db
            data_turn_iter['turn_num'] = turn_num
            data_iter['log'].append(copy.deepcopy(data_turn_iter))
            turn += 1
            turn_num += 1
        data[name] = data_iter

    return data