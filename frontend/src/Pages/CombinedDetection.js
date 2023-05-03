// import React, { useState ,useRef} from 'react';
// import { Cloudinary } from 'cloudinary-react';
// import { Button, CircularProgress, Container, Grid, Typography } from '@mui/material';

// const DetectionPage = () => {
//     const [videoFile, setVideoFile] = useState(null);
//     const [videoUrl, setVideoUrl] = useState(null);
//     const [fileName,setFileName]=useState("");
//     const [imageloading, setImageloading] = useState(false);
//     const [uploadError, setUploadError] = useState(null);
//     const [uploadSuccess, setUploadSuccess] = useState(false);
//     const [public_id,setPublic_id]=useState("")
//     const [detectError, setDetectError] = useState(null);
//     const [detectSuccess, setDetectSuccess] = useState(false);
//     const [detectloading, setDetectloading] = useState(false);
//     const [numberPlate,setNumberPlate]=useState("");

    
  
//     const handleFileChange = (e) => {
//       const file = e.target.files[0];
//       setVideoFile(event.target.files[0]);
//       setVideoUrl(URL.createObjectURL(event.target.files[0]));
//       setUploadSuccess(false)
//       setDetectSuccess(false)
//     };
  
//     const handleUpload = () => {
//       setImageloading(true);
//       setUploadError(null);
//       setUploadSuccess(false);
  
//       const formData = new FormData();
//       formData.append('video', videoFile);

  
//       fetch(`http://localhost:8000/uploadvideo`, {
//         method: 'POST',
//         body: formData,
//       })
//         .then((response) => response.json())
//         .then((data) => {
//           console.log(data.public_id);
//           setPublic_id(data.public_id)
//           setUploadSuccess(true);
//           setImageloading(false);
//         })
//         .catch((error) => {
//           console.error(error);
//           setUploadError(error);
//           setImageloading(false);
//         });
//     };

//     const handleDetection=()=>
//     {
//       setDetectloading(true);
//       setDetectError(null);
//       setDetectSuccess(false);
//       var detection_url="http://127.0.0.1:8000/image-detection?public_id="+public_id
//       fetch(detection_url, {
//         method: 'GET',
//       })
//         .then((response) => response.json())
//         .then((data) => {
//           console.log(data.message);
//           setNumberPlate(data.message)
//           setDetectSuccess(true)
//           setDetectloading(false);
//         })
//         .catch((error) => {
//           console.error(error);
//           setDetectError(error);
//           setDetectloading(false);
//         });
//     }
  
//     return (
//       <Container sx={{ mt: 2 }}>
//         <Grid container spacing={2} alignItems="center">
//           <Grid item xs={12}>
//             <input type="file" name="file" onChange={handleFileChange} />
//           </Grid>
//         </Grid>
//         {imageUrl && (
//           <Grid container sx={{ mt: 2 }}>
//             <Grid item xs={12}>
//               <img src={imageUrl} alt="Preview" style={{ maxWidth: '500px',maxHeight: '500px' }} />
//             </Grid>
//           </Grid>
//         )}
//         <Grid container sx={{ mt: 2 }} alignItems="center">
//           <Grid item xs={12}>
//             <Button variant="contained" disabled={!image || imageloading} onClick={handleUpload}>
//               Upload
//             </Button>
//           </Grid>
//         </Grid>
//         <Grid container sx={{ mt: 2 }} alignItems="center">
//           <Grid item xs={12}>
//             <Button variant="contained" disabled={!uploadSuccess || detectloading} onClick={handleDetection}>
//               Detect Number Plate
//             </Button>
//           </Grid>
//         </Grid>
//         {imageloading && (
//           <Grid container sx={{ mt: 2 }} alignItems="center">
//             <Grid item xs={12}>
//               <CircularProgress size={24} sx={{ mr: 1 }} />
//               <Typography variant="body1">Uploading image...</Typography>
//             </Grid>
//           </Grid>
//         )}
//         {uploadError && (
//           <Grid container sx={{ mt: 2 }} alignItems="center">
//             <Grid item xs={12}>
//               <Typography variant="body1" color="error">
//                 Upload error: {uploadError}
//               </Typography>
//             </Grid>
//           </Grid>
//         )}
//         {uploadSuccess && (
//           <Grid container sx={{ mt: 2 }} alignItems="center">
//             <Grid item xs={12}>
//               <Typography variant="body1" color="success">
//                 Upload successful!
//               </Typography>
//             </Grid>
//           </Grid>
//         )}
//         {detectloading && (
//           <Grid container sx={{ mt: 2 }} alignItems="center">
//             <Grid item xs={12}>
//               <CircularProgress size={24} sx={{ mr: 1 }} />
//               <Typography variant="body1"> Model is detecting the number plate ...</Typography>
//             </Grid>
//           </Grid>
//         )}
//         {detectError && (
//           <Grid container sx={{ mt: 2 }} alignItems="center">
//             <Grid item xs={12}>
//               <Typography variant="body1" color="error">
//                 Detection error: {detectError}
//               </Typography>
//             </Grid>
//           </Grid>
//         )}
//         {detectSuccess && (
//           <Grid container sx={{ mt: 2 }} alignItems="center">
//             <Grid item xs={12}>
//               <Typography variant="body1" color="success">
//                 Number Plate detected as {numberPlate}
//               </Typography>
//             </Grid>
//           </Grid>
//         )}
//       </Container>
//     );
//   };

// export default DetectionPage;
