# coding=gbk
all_domains = ['name', 'phone', 'id']
db_domains = []

slot_tag = {'B': 1, 'I': 2, 'O': 3}

requestable_slots = {
    "name": ["name"],
    "phone": ["phone"],
    "id": ["id"]
}

all_reqslot = ["name", "phone", "id"]

all_slots = all_reqslot
get_slot = {}
for s in all_slots:
    get_slot[s] = 1

dialog_acts = {
    "user": ['inform', 'update', 'other'],
    "name": ['request', 'continue', 'req_char', 'req_again', 'req_more', 'implicit_confirm', 'explicit_confirm', 'ack', 'req_correct',
             'compare', 'ask_restart', 'bye', 'good_signal', 'how_signal', 'bad_signal', 'repeat', 'robot', 'other', 'robot'],
    "car": ['robot', 'req_correct', 'ack', 'implicit_confirm', 'other', 'bye', 'req_more', 'how_signal', 'explicit_confirm', 'request', 'compare', 'continue']
}

all_acts = []
for acts in dialog_acts.values():
    for act in acts:
        if act not in all_acts:
            all_acts.append(act)

dialog_act_params = {
    "inform": all_slots + ['choice', 'open']
}

db_tokens = ['<sos_db>', '<eos_db>', '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]']

special_tokens = ['<pad>', '<go_r>', '<unk>', '<go_b>', '<go_a>', '<kd>',
                    '<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>', '<go_d>','<eos_c>',
                    '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_c>', '<sos_k>', '<eos_k>'] + db_tokens

eos_tokens = {
    'user': '<eos_u>', 'user_delex': '<eos_u>',
    'resp': '<eos_r>', 'resp_gen': '<eos_r>', 'pv_resp': '<eos_r>',
    'bspn': '<eos_b>', 'bspn_gen': '<eos_b>', 'pv_bspn': '<eos_b>',
    'bsdx': '<eos_b>', 'bsdx_gen': '<eos_b>', 'pv_bsdx': '<eos_b>',
    'aspn': '<eos_a>', 'aspn_gen': '<eos_a>', 'pv_aspn': '<eos_a>',
    'dspn': '<eos_d>', 'dspn_gen': '<eos_d>', 'pv_dspn': '<eos_d>',
    'cc': '<eos_c>', 'cc_gen': '<eos_c>',
    'kdpn': '<eos_k>', 'kdpn_gen': '<eos_k>', 'pv_kdpn': '<eos_k>'}

sos_tokens = {
    'user': '<sos_u>', 'user_delex': '<sos_u>',
    'resp': '<sos_r>', 'resp_gen': '<sos_r>', 'pv_resp': '<sos_r>',
    'bspn': '<sos_b>', 'bspn_gen': '<sos_b>', 'pv_bspn': '<sos_b>',
    'bsdx': '<sos_b>', 'bsdx_gen': '<sos_b>', 'pv_bsdx': '<sos_b>',
    'aspn': '<sos_a>', 'aspn_gen': '<sos_a>', 'pv_aspn': '<sos_a>',
    'dspn': '<sos_d>', 'dspn_gen': '<sos_d>', 'pv_dspn': '<sos_d>',
    'cc': '<sos_c>', 'cc_gen': '<sos_c>',
    'kdpn': '<sos_k>', 'kdpn_gen': '<sos_k>', 'pv_kdpn': '<sos_k>'}