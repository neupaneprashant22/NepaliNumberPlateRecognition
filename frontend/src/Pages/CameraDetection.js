import React, { useState, useEffect, useRef } from 'react';

const CameraCapture = () => {
  const [imageSrc, setImageSrc] = useState(null);
  const [videoStream, setVideoStream] = useState(null);
  const videoRef = useRef(null);

  const handleStartCaptureClick = () => {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      setVideoStream(stream);
      videoRef.current.srcObject = stream;
      videoRef.current.play();
    });
  };

  const handleStopCaptureClick = () => {
    videoStream.getTracks().forEach((track) => track.stop());
    setVideoStream(null);
  };

  const handleCaptureClick = () => {
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(videoRef.current, 0, 0);
    setImageSrc(canvas.toDataURL('image/png'));
  };

  useEffect(() => {
    return () => {
      if (videoStream) {
        videoStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [videoStream]);

  return (
    <div>
      <video ref={videoRef} width="400" height="300" />
      <br />
      <button onClick={handleStartCaptureClick}>Start Capture</button>
      <button onClick={handleStopCaptureClick}>Stop Capture</button>
      <button onClick={handleCaptureClick}>Capture</button>
      {imageSrc && <img src={imageSrc} width="400" height="300" />}
    </div>
  );
};

export default CameraCapture;
