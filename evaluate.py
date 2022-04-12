def validate_metric(results):
    joint_acc = 0.0
    slot_acc = 0.0
    dialog_succ = 0.0
    total_turn_num = 0.0
    for name, dialog in results.items():
        for turn in dialog['turns']:
            bs_target = turn['bs']
            bs_gen = turn['bs_gen']
            joint_acc += bs_gen == bs_target

            bs_gen = list(filter(lambda x:x!='', bs_gen.split(',')))
            bs_target = bs_target.split(',')
            iter_slot_acc = 0
            for i in range(len(bs_target)):
                if i < len(bs_gen):
                    iter_slot_acc += bs_gen[i] == bs_target[i]
                else:
                    continue
            if len(bs_target) > 0:
                slot_acc += iter_slot_acc / len(bs_target)
            else:
                slot_acc += 1 if len(bs_gen) == 0 else 0
            total_turn_num += 1

        dialog_succ += ''.join(bs_gen).replace(',', '') == dialog['goal']
    return joint_acc/total_turn_num, slot_acc/total_turn_num, dialog_succ/len(results)

def validation_metric_gpt(data_path, dataloader, results):
    data = dataloader.read_data(data_path)
    # block acc, state acc, success
    total_turn_num = 0
    block_acc = 0
    slot_acc = 0
    success = 0
    total_data = 0
    for turn in results:
        dial_id = turn['dial_id']
        last_turn_num = len(data[dial_id]['log']) - 1
        turn_num = turn['turn_num']
        total_turn_num += 1
        state_gen = turn['bspn_gen']
        state_label = turn['bspn']
        block_acc += (state_gen == state_label)
        state_gen = state_gen.split(',')
        state_label = state_label.split(',')
        iter_slot_acc = 0
        for i in range(len(state_label)):
            if i < len(state_gen):
                iter_slot_acc += state_gen[i] == state_label[i]
            else:
                continue
        if len(state_label) > 0:
            slot_acc += iter_slot_acc / len(state_label)
        else:
            slot_acc += 1 if len(state_gen) == 0 else 0

        if turn_num == last_turn_num:
            total_data += 1
            goal = data[dial_id]['goal']
            if ''.join(state_gen).replace(',', '') == goal:
                success += 1
    return block_acc/total_turn_num, slot_acc/total_turn_num, success/total_data