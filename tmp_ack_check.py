"""Печать ACK, который наш патч строит для ответа 200 OK от Asterisk."""

from types import SimpleNamespace

import pyVoIP.VoIP  # noqa: F401
from pyVoIP.SIP import SIPClient, SIPMessage

from sipbot import pyvoip_fixes

pyvoip_fixes.install()

OK_200 = (
    b"SIP/2.0 200 OK\r\n"
    b"Via: SIP/2.0/UDP 172.18.185.162:5060;branch=z9hG4bK0d6b0c4e"
    b";received=172.18.185.162;rport=5060\r\n"
    b'From: <sip:503@172.18.185.162>;tag=7f3c1a2b\r\n'
    b"To: <sip:+79780481885@sip.evpanet.com>;tag=as5f3a2b1c\r\n"
    b"Call-ID: 4b227777d4dd1fc61c6f884f48641d02@172.18.185.162:5060\r\n"
    b"CSeq: 2 INVITE\r\n"
    b"Contact: <sip:+79780481885@91.195.230.4:5060>\r\n"
    b"Content-Type: application/sdp\r\n"
    b"Content-Length: 0\r\n\r\n"
)

client = SIPClient(
    "sip.evpanet.com",
    5060,
    "503",
    "secret",
    SimpleNamespace(_status=None, calls={}),  # type: ignore[arg-type]
    myIP="172.18.185.162",
    myPort=5060,
)
message = SIPMessage(OK_200)
client.tagLibrary[message.headers["Call-ID"]] = "7f3c1a2b"
print(client.gen_ack(message))
