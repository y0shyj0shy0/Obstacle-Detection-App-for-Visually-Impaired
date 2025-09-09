import React, { useEffect, useRef, useState } from 'react';
import { View, StyleSheet, Text, Alert } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import axios from 'axios';
import Tts from 'react-native-tts';
const App = () => {
  const [hasPermission, setHasPermission] = useState(false);
  const [hello, setHello] = useState('');
  const device = useCameraDevice('back');
  const camera = useRef<Camera>(null);
  const [isCapturing, setIsCapturing] = useState(false); // Capturing 상태 관리
  const [isDeviceReady, setIsDeviceReady] = useState(false); // Device 준비 상태
  const [ttstext, setTtsText] = useState('');
  var temp;
  // 카메라 권한 요청
  useEffect(() => {
    const requestPermission = async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'granted');
    };
    requestPermission();

  }, []);

  // 서버 데이터 요청
  useEffect(() => {
    axios.get('http://59.19.113.82:3003/')
      .then((res) => {
        console.log(res.data);
        setHello(res.data.d);
      })
      .catch((err) => {
        console.error(err);
      });
  }, []);

  // device 초기화 체크
  useEffect(() => {
    if (device) {
      setIsDeviceReady(true);
    }
  }, [device]);

  // 매 프레임마다 사진을 찍고 서버로 전송
  useEffect(() => {
    let intervalId: NodeJS.Timer | null = null;

    const captureAndSend = async () => {
      if (camera.current) {
        try {
          const photo = await camera.current.takePhoto({
            qualityPrioritization: 'quality', // 사진 품질 우선
          });

          console.log('Captured photo:', photo.path);

          const formData = new FormData();
          formData.append('file', {
            uri: `file://${photo.path}`,
            type: 'image/jpeg',
            name: 'frame.jpg',
          });

          const response = await axios.post('http://59.19.113.82:3003/upload', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });

          const getTTS = await axios.get('http://59.19.113.82:3003/getTTS');
          console.log('TTS data: ', getTTS.data);
          if (getTTS.data) {
            Tts.speak(getTTS.data.tts);

          }
          console.log('Server response:', response.data);
        } catch (error) {
          console.error('Error capturing and sending photo:', error);
        }
      }
    };

    if (isCapturing) {
      // 30 FPS 기준으로 33ms마다 실행
      intervalId = setInterval(captureAndSend, 1000 / 10);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isCapturing]);

  // 카메라 준비 중 표시
  if (!isDeviceReady) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>Loading Camera...</Text>
      </View>
    );
  }

  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>
          Camera permission is required.
        </Text>
        <Text onPress={() => Camera.requestCameraPermission().then((status) => {
          if (status === 'granted') setHasPermission(true);
          else Alert.alert('Permission Denied');
        })}>
          Grant Permission
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        photo={true} // 사진 촬영 가능
      />
      <Text
        style={styles.captureText}
        onPress={() => setIsCapturing((prev) => !prev)}
      >
        {isCapturing ? 'Stop Capture' : 'Start Capture'}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  permissionText: {
    color: '#fff',
    fontSize: 16,
    marginBottom: 20,
  },
  captureText: {
    color: '#fff',
    fontSize: 18,
    marginTop: 20,
    padding: 10,
    backgroundColor: 'blue',
    borderRadius: 5,
    textAlign: 'center',
  },
});

export default App;


//ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

// import React, { useEffect, useRef, useState } from 'react';
// import { View, StyleSheet, Text, Button, Alert } from 'react-native';
// import { Camera, useCameraDevice } from 'react-native-vision-camera';
// import Webcam from "react-webcam";
// import axios from 'axios';

// const App = () => {
//   const videoConstraints = {
//     width: 1280,
//     height: 720,
//     facingMode: "user"
//   };
  
//   const WebcamStreamCapture = () => {
//     const webcamRef = React.useRef(null);
//     const mediaRecorderRef = React.useRef(null);
//     const [capturing, setCapturing] = React.useState(false);
//     const [recordedChunks, setRecordedChunks] = React.useState([]);
  
//     const handleStartCaptureClick = React.useCallback(() => {
//       setCapturing(true);
//       mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
//         mimeType: "video/webm"
//       });
//       mediaRecorderRef.current.addEventListener(
//         "dataavailable",
//         handleDataAvailable
//       );
//       mediaRecorderRef.current.start();
//     }, [webcamRef, setCapturing, mediaRecorderRef]);
  
//     const handleDataAvailable = React.useCallback(
//       ({ data }) => {
//         if (data.size > 0) {
//           setRecordedChunks((prev) => prev.concat(data));
//         }
//       },
//       [setRecordedChunks]
//     );
  
//     const handleStopCaptureClick = React.useCallback(() => {
//       mediaRecorderRef.current.stop();
//       setCapturing(false);
//     }, [mediaRecorderRef, webcamRef, setCapturing]);
  
//     const handleDownload = React.useCallback(() => {
//       if (recordedChunks.length) {
//         const blob = new Blob(recordedChunks, {
//           type: "video/webm"
//         });
//         const url = URL.createObjectURL(blob);
//         const a = document.createElement("a");
//         document.body.appendChild(a);
//         a.style = "display: none";
//         a.href = url;
//         a.download = "react-webcam-stream-capture.webm";
//         a.click();
//         window.URL.revokeObjectURL(url);
//         setRecordedChunks([]);
//       }
//     }, [recordedChunks]);
  
//     return (
//       <>
//         <Webcam audio={false} ref={webcamRef} />
//         {capturing ? (
//           <button onClick={handleStopCaptureClick}>Stop Capture</button>
//         ) : (
//           <button onClick={handleStartCaptureClick}>Start Capture</button>
//         )}
//         {recordedChunks.length > 0 && (
//           <button onClick={handleDownload}>Download</button>
//         )}
//       </>
//     );
//   };
  
//   // ReactDOM.render(<WebcamStreamCapture />, document.getElementById("root"));
// };

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     backgroundColor: '#000',
//     justifyContent: 'center',
//     alignItems: 'center',
//   },
//   permissionText: {
//     color: '#fff',
//     fontSize: 16,
//     marginBottom: 20,
//   },
// });

// export default App;



