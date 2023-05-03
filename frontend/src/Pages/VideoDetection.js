import React, { useState } from 'react';
import axios from 'axios';
import { Box, Button, CircularProgress, Grid, Paper, Typography } from '@mui/material';
import { List, ListItem, ListItemText } from "@mui/material";
import { Document, Page, Text, View, PDFDownloadLink,StyleSheet } from '@react-pdf/renderer';

function VideoDetection() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [fileName,setFileName]=useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [detectionResult, setDetectionResult] = useState("");
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [detectError, setDetectError] = useState(null);
  const [detectSuccess, setDetectSuccess] = useState(false);
  const [detectloading, setDetectloading] = useState(false);
  const [imageloading,setImageloading]=useState(false)
  const [formattedOutput,setFormattedOutput]=useState([]);

  const currentDate = new Date().toLocaleString();

  const styles = StyleSheet.create({
    listContainer: {
      marginTop: 10,
      marginLeft: 20,
      marginRight: 20,
      fontSize: 12,
    },
    listItem: {
      marginBottom: 5,
    },
    listItemText: {
      marginLeft: 5,
    },
    button: {
      padding: 10,
      backgroundColor: "#007bff",
      color: "#fff",
      borderRadius: 5,
      textAlign: "center",
      textDecoration: "none",
      display: "inline-block",
      margin: "10px",
      fontSize: 16,
      fontWeight: "bold",
    },
  });

  const MyDocument = () => (
    <Document>
      <Page>
        <View>
          <Text>The number plates detected on {currentDate} are:</Text>
          <View style={styles.listContainer}>
            {formattedOutput.map((plate, index) => (
              <View key={index} style={styles.listItem}>
                <Text>{index + 1}. <Text style={styles.listItemText}>{plate}</Text></Text>
              </View>
            ))}
          </View>
        </View>
      </Page>
    </Document>
  );

  const handleOutputFormat=(numberplates)=>
  {
    console.log((numberplates))
    numberplates=String(numberplates)
    var arrayofnumberplates=numberplates.split(",")
    for (let i = 0; i < arrayofnumberplates.length; i++) {
      if ((arrayofnumberplates[i] == '0')|| (arrayofnumberplates[i] == ' ')) {
        arrayofnumberplates.splice(i, 1);
        i--; // adjust the index to account for the removed element
      }
    }
    arrayofnumberplates = [...new Set(arrayofnumberplates)];
    setFormattedOutput(arrayofnumberplates)
    console.log(arrayofnumberplates)
  }


  const handleFileChange = (event) => {
    setVideoFile(event.target.files[0]);
    setVideoUrl(URL.createObjectURL(event.target.files[0]));
    setUploadSuccess(false)
    setDetectSuccess(false)
  };

  const handleUpload = async () => {
    setImageloading(true);
    setUploadError(null);
    setUploadSuccess(false);
    const formData = new FormData();
    formData.append('video', videoFile);
    try {
      const response = await axios.post('http://localhost:8000/uploadvideo/', formData);
      setFileName(response.data);
      setUploadSuccess(true);
      setImageloading(false);
    } catch (error) {
      console.error(error);
      setUploadSuccess(error);
      setImageloading(false);
    }
  };

  const handleDetect = async () => {
    setDetectloading(true);
    setDetectError(null);
    setDetectSuccess(false);
    try {
      const response = await axios.get(`http://localhost:8000/video-detection?video_url=${fileName}`);
      var result=(response.data.message)
      console.log((result))
      handleOutputFormat(result);
      setDetectionResult(result);
      setDetectSuccess(true)
      setDetectloading(false);
    } catch (error) {
      setDetectError(error);
      setDetectloading(false);
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <Box sx={{ m: 2 }}>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <Typography variant="h5">Upload and Detect Video</Typography>
        </Grid>
        <Grid item xs={12} sm={6}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <input type="file" accept="video/*" onChange={handleFileChange} />
              {videoUrl && (
                <Box mt={2}>
                  <video src={videoUrl} controls width="400" height="400" />
                </Box>
              )}
            </Box>
            <Button variant="contained" onClick={handleUpload} disabled={!videoFile}>
                {loading ? <CircularProgress size={24} /> : "Upload"}
              </Button>
              {success && <Typography color="green">Video uploaded successfully</Typography>}
              <Box mt={2}>
                <Button variant="contained" onClick={handleDetect} disabled={!fileName}>
                  {loading ? <CircularProgress size={24} /> : "Detect"}
                </Button>
              </Box>
          </Paper>
        </Grid>
        <Grid item xs={12} sm={6}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              {success && detectionResult && <Typography mt={2}>{detectionResult}</Typography>}
            </Box>
            {imageloading && (
          <Grid container sx={{ mt: 2 }} alignItems="center">
            <Grid item xs={12}>
              <CircularProgress size={24} sx={{ mr: 1 }} />
              <Typography variant="body1">Uploading image...</Typography>
            </Grid>
          </Grid>
        )}
        {uploadError && (
          <Grid container sx={{ mt: 2 }} alignItems="center">
            <Grid item xs={12}>
              <Typography variant="body1" color="error">
                Upload error: {uploadError}
              </Typography>
            </Grid>
          </Grid>
        )}
        {uploadSuccess && (
          <Grid container sx={{ mt: 2 }} alignItems="center">
            <Grid item xs={12}>
              <Typography variant="body1" color="success">
                Upload successful!
              </Typography>
            </Grid>
          </Grid>
        )}
        {detectloading && (
          <Grid container sx={{ mt: 2 }} alignItems="center">
            <Grid item xs={12}>
              <CircularProgress size={24} sx={{ mr: 1 }} />
              <Typography variant="body1"> Model is detecting the number plate ...</Typography>
            </Grid>
          </Grid>
        )}
        {detectError && (
          <Grid container sx={{ mt: 2 }} alignItems="center">
            <Grid item xs={12}>
              <Typography variant="body1" color="error">
                Detection error: {detectError}
              </Typography>
            </Grid>
          </Grid>
        )}
        {detectSuccess && (
          <Grid container sx={{ mt: 2 }} alignItems="center">
            <Grid item xs={12}>
              <Typography variant="body1" color="success">
                {/* Number Plate detected as {detectionResult} */}
                <List>
                  The Number Plates detected in the video are:
                  {formattedOutput.map((item, index) => (
                    <ListItem key={index}>
                    <ListItemText
                      disableTypography
                      primary={<Typography variant="body2" style={{ color: '#000',fontSize:'20px' }}>{index + 1}. {item}</Typography>}
                    />
                  </ListItem>
                  ))}
                </List>
              </Typography>
              <a href="#" style={styles.button}>
              <PDFDownloadLink document={<MyDocument />} fileName="DetectedPlates.pdf">
                <Typography style={{ color: '#fff',fontSize:'15px' }}>Download PDF</Typography>
              </PDFDownloadLink>
              </a>
            </Grid>
          </Grid>
        )}
          </Paper>
        </Grid>
      </Grid>
      {/* {detectSuccess &&(
      <PDFDownloadLink document={<MyDocument />} fileName="DetectedPlates.pdf">
        Download PDF
      </PDFDownloadLink>
      )} */}
    </Box>
    
  );
}

export default VideoDetection;
