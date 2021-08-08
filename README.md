# Installation

>`pip install -r requirements.txt`

# Instructions

 - all columns have been used to train the model. `Test data f1-score is 99.9%`

 - Inside the API folder run (on CMD):

> `python manage.py runserver`

this will start server at : `http://127.0.0.1:8000`



 - send a **get** request to  `http://127.0.0.1:8000/predict/` (you can use postman for testing)

sample JSON request:
>`{
"service":"-",
"state":"FIN",
"proto":"tcp",
"attack_cat":"Normal",
"ID_NO":12,
"dur":100,
"spkts":10,
"dpkts":10,
"sbytes":100,
"dbytes":100,
"rate":100,
"sttl":100,
"dttl":100,
"sload":12,
"dload":12,
"sloss":12,
"dloss":12,
"sinpkt":15,
"dinpkt":15,
"sjit":15,
"djit":15,
"swin":15,
"stcpb":48,
"dtcpb":48,
"dwin":48,
"tcprtt":98,
"synack":98,
"ackdat":98,
"smean":98,
"dmean":98,
"trans_depth":8,
"response_body_len":54,
"ct_srv_src":56,
"ct_state_ttl":55,
"ct_dst_ltm":55,
"ct_src_dport_ltm":87.215,
"ct_dst_sport_ltm":87.215,
"ct_dst_src_ltm":87.215,
"is_ftp_login":87.215,
"ct_ftp_cmd":87.215,
"ct_flw_http_mthd":0.225,
"ct_src_ltm":0.225,
"ct_srv_dst":0.225,
"is_sm_ips_ports":1.25
}`



 - API sends a response :
> `{
    "pred_probability": 0.0,
    "input_request": {
        "service": "-",
        "state": "FIN",
        "proto": "tcp",
        "attack_cat": "Normal",
        "ID_NO": 12,
        "dur": 100,
        "spkts": 10,
        "dpkts": 10,
        "sbytes": 100,
        "dbytes": 100,
        "rate": 100,
        "sttl": 100,
        "dttl": 100,
        "sload": 12,
        "dload": 12,
        "sloss": 12,
        "dloss": 12,
        "sinpkt": 15,
        "dinpkt": 15,
        "sjit": 15,
        "djit": 15,
        "swin": 15,
        "stcpb": 48,
        "dtcpb": 48,
        "dwin": 48,
        "tcprtt": 98,
        "synack": 98,
        "ackdat": 98,
        "smean": 98,
        "dmean": 98,
        "trans_depth": 8,
        "response_body_len": 54,
        "ct_srv_src": 56,
        "ct_state_ttl": 55,
        "ct_dst_ltm": 55,
        "ct_src_dport_ltm": 87.215,
        "ct_dst_sport_ltm": 87.215,
        "ct_dst_src_ltm": 87.215,
        "is_ftp_login": 87.215,
        "ct_ftp_cmd": 87.215,
        "ct_flw_http_mthd": 0.225,
        "ct_src_ltm": 0.225,
        "ct_srv_dst": 0.225,
        "is_sm_ips_ports": 1.25
    }
}`

> `pred_probability` gives the predicted probability in range 0.0 --> 1.0.   
