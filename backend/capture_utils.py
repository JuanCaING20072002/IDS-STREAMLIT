import shlex
from subprocess import PIPE, Popen
import pandas as pd


def build_tshark_command() -> list:
    comm_arg = (
        "sudo /usr/local/bin/tshark "
        "-i eth0 -i wlan0 "
        "-l "
        "-Y 'eth.type != 0x8899' "
        "-f \"tcp or udp or arp\" "
        "-f \"arp or icmp or (udp and not port 53 and not port 5353) or (tcp and not port 443)\" "
        "-T fields -E separator=/t "
        "-e frame.time_delta "
        "-e _ws.col.Protocol "
        "-e ip.src -e ip.dst "
        "-e arp.src.proto_ipv4 -e arp.dst.proto_ipv4 "
        "-e ipv6.src -e ipv6.dst "
        "-e eth.src -e eth.dst "
        "-e tcp.srcport -e tcp.dstport "
        "-e udp.srcport -e udp.dstport "
        "-e frame.len -e udp.length "
        "-e ip.ttl -e icmp.type "
        "-e ip.dsfield.dscp -e ip.flags.rb -e ip.flags.df -e ip.flags.mf "
        "-e tcp.flags.res -e tcp.flags.ns -e tcp.flags.cwr -e tcp.flags.ecn "
        "-e tcp.flags.urg -e tcp.flags.ack -e tcp.flags.push -e tcp.flags.reset "
        "-e tcp.flags.syn -e tcp.flags.fin -e ip.version "
        "-e frame.time_epoch"
    )
    return shlex.split(comm_arg)


def capturing_packets(comm_arg):
    process = Popen(comm_arg, stdout=PIPE, stderr=PIPE, text=True)
    return process


def type_packet(packet_):
    if len(packet_) < 34:  # Rellenar con valores vacíos si falta información
        packet_ += [''] * (34 - len(packet_))

    if packet_[1] in ['TCP', 'SSH', 'SSHv2', 'TLSv1.2', 'HTTP']:
        tcp_src = packet_[10]
        tcp_dst = packet_[11]
    elif packet_[1] in ['UDP', 'DNS', 'BROWSER', 'DHCP']:
        tcp_src = packet_[12]
        tcp_dst = packet_[13]
    else:
        tcp_src = ''
        tcp_dst = ''

    if packet_[1] == 'ARP':
        packet_[2] = packet_[4]  # ip.src <- arp.src.proto_ipv4
        packet_[3] = packet_[5]  # ip.dst <- arp.dst.proto_ipv4

    if packet_[32] == '6':
        packet_[2] = packet_[6]  # ip.src <- ipv6.src
        packet_[3] = packet_[7]  # ip.dst <- ipv6.dst

    for index in [16, 18, 19, 20, 21]:
        if packet_[index]:
            packet_[index] = packet_[index].replace(',', '.')

    ordered_packet = [
        packet_[0], packet_[1], packet_[2], packet_[3], tcp_src, tcp_dst, packet_[14], packet_[15],
        packet_[16], packet_[17], packet_[18], packet_[19], packet_[20], packet_[21], packet_[22],
        packet_[23], packet_[24], packet_[25], packet_[26], packet_[27], packet_[28], packet_[29],
        packet_[30], packet_[31], packet_[33]
    ]

    fieldnames = [
        'delta_time', 'protocols','ip_src','ip_dst','port_src', 'port_dst', 'frame_len',
        'udp_len', 'ip_ttl', 'icmp_type', 'tos', 'ip_flags_rb', 'ip_flags_df', 'ip_flags_mf',
        'tcp_flags_res', 'tcp_flags_ns', 'tcp_flags_cwr', 'tcp_flags_ecn', 'tcp_flags_urg',
        'tcp_flags_ack', 'tcp_flags_push', 'tcp_flags_reset', 'tcp_flags_syn', 'tcp_flags_fin','epoch_time'
    ]

    return {fieldnames[i]: [ordered_packet[i]] for i in range(len(fieldnames))}


def packet_df(type_packet_dict, df):
    df_temp = pd.DataFrame(type_packet_dict)
    if list(df_temp.columns) == ['value']:
        return df  # Ignorar tablas value
    if df is None:
        df = df_temp
    else:
        df = pd.concat([df, df_temp], axis=0, ignore_index=True)
    if list(df.columns) == ['value']:
        return None
    return df
