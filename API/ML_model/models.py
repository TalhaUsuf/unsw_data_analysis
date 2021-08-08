from django.db import models

# Create your models here.
#  all things related to database

class inputs(models.Model):

    '''
    proto              82332 non-null  object
 3   service            82332 non-null  object
 4   state              82332 non-null  object
     attack_cat         82332 non-null  object
    '''
    attack_cat = models.CharField(max_length=100)
    service = models.CharField(max_length=100)
    proto = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    ID_NO = models.FloatField()
    dur = models.FloatField()
    spkts = models.FloatField()
    dpkts = models.FloatField()
    sbytes = models.FloatField()
    dbytes = models.FloatField()
    rate = models.FloatField()
    sttl = models.FloatField()
    dttl = models.FloatField()
    sload = models.FloatField()
    dload = models.FloatField()
    sloss = models.FloatField()
    dloss = models.FloatField()
    sinpkt = models.FloatField()
    dinpkt = models.FloatField()
    sjit = models.FloatField()
    djit = models.FloatField()
    swin = models.FloatField()
    stcpb = models.FloatField()
    dtcpb = models.FloatField()
    dwin  = models.FloatField()
    tcprtt = models.FloatField()
    synack = models.FloatField()
    ackdat = models.FloatField()
    smean = models.FloatField()
    dmean = models.FloatField()
    trans_depth = models.FloatField()
    response_body_len = models.FloatField()
    ct_srv_src = models.FloatField()
    ct_state_ttl = models.FloatField()
    ct_dst_ltm = models.FloatField()
    ct_src_dport_ltm = models.FloatField()
    ct_dst_sport_ltm = models.FloatField()
    ct_dst_src_ltm = models.FloatField()
    is_ftp_login = models.FloatField()
    ct_ftp_cmd = models.FloatField()
    ct_flw_http_mthd = models.FloatField()
    ct_src_ltm = models.FloatField()
    ct_srv_dst = models.FloatField()
    is_sm_ips_ports = models.FloatField()
