import logo from './logo.svg';
import './App.css';
import Main from './Pages/main';
import LandingPage from './Pages/LandingPage';
import { BrowserRouter } from 'react-router-dom';
import { Route } from 'react-router-dom';
import { Routes } from 'react-router-dom';
import DetectionPage from './Pages/DetectionPage';
import CameraCapture from './Pages/CameraDetection'
import CombinedDetectionPage from './Pages/CombinedDetection';
import VideoDetection from './Pages/VideoDetection';

function App() {
  return (
    <BrowserRouter>
    <Routes>
      <Route path='/' element={<LandingPage/>}></Route>
      <Route path='/detection' element={<DetectionPage/>}></Route>
      <Route path='/videodetection' element={<VideoDetection/>}></Route>
      <Route path='/capture' element={<CameraCapture/>}></Route>
    </Routes>
    </BrowserRouter>
  );
}

export default App;
