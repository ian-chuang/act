from google.cloud import firestore
import json
import asyncio
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc import VideoStreamTrack
from av import VideoFrame
import numpy as np
from aiortc.rtcrtpsender import RTCRtpSender
import time

class BufferVideoStreamTrack(VideoStreamTrack):
    def __init__(self, buffer_size=1, image_format="rgb24", fps="24"):
        super().__init__()
        self.queue = asyncio.Queue(maxsize=buffer_size)
        self.image_format = image_format
        self.fps = fps

    async def recv(self):
        start_time = time.time()
        pts, time_base = await self.next_timestamp()
        frame = await self.queue.get()
        frame = VideoFrame.from_ndarray(frame, format=self.image_format)
        frame.pts = pts
        frame.time_base = time_base
        end_time = time.time()
        # rate limit 
        await asyncio.sleep(max(0, 1/float(self.fps) - (end_time - start_time)))
        return frame

    def add_frame(self, frame):
        try:
            self.queue.put_nowait(frame)
        except asyncio.QueueFull:
            # print("[BufferVideoStreamTrack] Frame queue is full")
            pass

def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )
        

class WebRTCHeadset:
    def __init__(
        self,
        loop,
        serviceAccountKeyFile='serviceAccountKey.json',
        signalingSettingsFile='signalingSettings.json',
        video_buffer_size=1,
        data_buffer_size=1,
    ):
        self.loop = loop

        with open(serviceAccountKeyFile) as f:
            serviceAccountKey = json.load(f)

        self.db = firestore.Client.from_service_account_info(serviceAccountKey)

        with open(signalingSettingsFile) as f:
            self.signalingSettings = json.load(f)

        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration([
                RTCIceServer("stun:stun1.l.google.com:19302"),
                RTCIceServer("stun:stun2.l.google.com:19302"),
                RTCIceServer(self.signalingSettings['turn_server_url'], self.signalingSettings['turn_server_username'], self.signalingSettings['turn_server_password'])            
            ])
        )
        self.robotId = self.signalingSettings['robotID']
        self.password = self.signalingSettings['password']

        self.channel = None
        self.video_track = None
        self.video_buffer_size = video_buffer_size

        self.receive_data_queue = asyncio.Queue(maxsize=data_buffer_size)
        self.send_data_queue = asyncio.Queue(maxsize=data_buffer_size)

        # create send timer
        async def send_data():
            while True:
                try:
                    if self.channel is not None:
                        data = await self.send_data_queue.get()
                        self.channel.send(data)
                        await asyncio.sleep(1/10)
                except Exception as e:
                    print(f"Failed to send data: {e}")
                    await asyncio.sleep(1)

        self.loop.create_task(send_data())

    def receive_data(self):
        try:
            data = self.receive_data_queue.get_nowait()
            return data
        except asyncio.QueueEmpty:
            return None
    
    def send_image(self, image):
        try:
            if self.video_track is not None:
                self.video_track.add_frame(image)
        except Exception as e:
            print(f"Failed to send image: {e}")

    def send_data(self, data):
        try:
            self.send_data_queue.put_nowait(json.dumps(data))
        except asyncio.QueueFull:
            pass
            #print("[RobotWebRTC] Data queue is full")

    async def run_offer(self):
        print("running offer")
        self.channel = self.pc.createDataChannel("control")

        @self.channel.on("open")
        def on_open():
            print("channel open")    

        self.video_track = BufferVideoStreamTrack(buffer_size=self.video_buffer_size, image_format="rgb24")
        self.video_sender = self.pc.addTrack(self.video_track)
        force_codec(self.pc, self.video_sender, 'video/h264')
            
        call_doc = self.db.collection(self.password).document(self.robotId)

        # send offer
        await self.pc.setLocalDescription(await self.pc.createOffer())
        call_doc.set(
            {
                'sdp': self.pc.localDescription.sdp,
                'type': self.pc.localDescription.type
            }
        )

        future = asyncio.Future()
        def answer_callback(doc_snapshot, changes, read_time):
            for doc in doc_snapshot:
                if self.pc.remoteDescription is None and doc.to_dict()['type'] == 'answer':
                    data = doc.to_dict()
                    self.loop.call_soon_threadsafe(future.set_result, data)
        doc_watch = call_doc.on_snapshot(answer_callback)
        print('waiting for answer...')
        data = await future
        doc_watch.unsubscribe()

        await self.pc.setRemoteDescription(RTCSessionDescription(
            sdp=data['sdp'],
            type=data['type']
        ))

        # delete call document
        call_doc = self.db.collection(self.password).document(self.robotId)
        call_doc.delete()

        # add event listener for connection close
        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            if self.pc.iceConnectionState == "closed":
                print("Connection closed, restarting...")
                await self.restart_connection()

    async def restart_connection(self):
        # close current peer connection
        await self.pc.close()

        # create new peer connection
        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration([
                RTCIceServer("stun:stun1.l.google.com:19302"),
                RTCIceServer("stun:stun2.l.google.com:19302"),
                RTCIceServer(self.signalingSettings['turn_server_url'], self.signalingSettings['turn_server_username'], self.signalingSettings['turn_server_password'])
            ])
        )

        # run offer again
        await self.run_offer()

    

async def main():
    headset = WebRTCHeadset(asyncio.get_event_loop())
    await headset.run_offer()

    while True:
        if headset.receive_data():
            print("Received headset data")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())