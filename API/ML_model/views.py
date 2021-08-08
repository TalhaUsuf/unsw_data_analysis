from django.shortcuts import render
from rest_framework.views import APIView
# Create your views here.
from . apps import MLModelConfig
from django.http import JsonResponse
import joblib
import pandas as pd
from . models import inputs
from rest_framework import serializers


# validate the input request

class validate(serializers.Serializer):
    proto = serializers.CharField(max_length=100, allow_blank=False)
    state = serializers.CharField(max_length=100, allow_blank=False)
    attack_cat = serializers.CharField(max_length=100, allow_blank=False)
    service = serializers.CharField(max_length=100, allow_blank=False)
    ID_NO = serializers.FloatField()
    dur = serializers.FloatField()
    spkts = serializers.FloatField()
    dpkts = serializers.FloatField()
    sbytes = serializers.FloatField()
    dbytes = serializers.FloatField()
    rate = serializers.FloatField()
    sttl = serializers.FloatField()
    dttl = serializers.FloatField()
    sload = serializers.FloatField()
    dload = serializers.FloatField()
    sloss = serializers.FloatField()
    dloss = serializers.FloatField()
    sinpkt = serializers.FloatField()
    dinpkt = serializers.FloatField()
    sjit = serializers.FloatField()
    djit = serializers.FloatField()
    swin = serializers.FloatField()
    stcpb = serializers.FloatField()
    dtcpb = serializers.FloatField()
    dwin = serializers.FloatField()
    tcprtt = serializers.FloatField()
    synack = serializers.FloatField()
    ackdat = serializers.FloatField()
    smean = serializers.FloatField()
    dmean = serializers.FloatField()
    trans_depth = serializers.FloatField()
    response_body_len = serializers.FloatField()
    ct_srv_src = serializers.FloatField()
    ct_state_ttl = serializers.FloatField()
    ct_dst_ltm = serializers.FloatField()
    ct_src_dport_ltm = serializers.FloatField()
    ct_dst_sport_ltm = serializers.FloatField()
    ct_dst_src_ltm = serializers.FloatField()
    is_ftp_login = serializers.FloatField()
    ct_ftp_cmd = serializers.FloatField()
    ct_flw_http_mthd = serializers.FloatField()
    ct_src_ltm = serializers.FloatField()
    ct_srv_dst = serializers.FloatField()
    is_sm_ips_ports = serializers.FloatField()


class predict(APIView):


    def get(self, request):
        if request.method == 'GET':
            flag = validate(data=request.data)
            if flag.is_valid():
                # if True:

                params = request.data

                # convert to dataframe
                p = {}
                for k,v in params.items():
                    p.setdefault(f'{k}',[]).append(v)
                df = pd.DataFrame.from_dict(p) # doesnot contain label field


                out = MLModelConfig.preprocess.transform(df)
                out = MLModelConfig.model.predict(out).squeeze().tolist()
                response = {
                            "pred_probability" : out,
                            "input_request" : params,}
                return JsonResponse(response)