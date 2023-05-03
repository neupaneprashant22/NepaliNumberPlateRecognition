import React, { useState ,useRef} from 'react';
import { Cloudinary } from 'cloudinary-react';
import { Box, Button, CircularProgress, Container, Grid, Typography,Paper } from '@mui/material';
import { List, ListItem, ListItemText } from "@mui/material";

const DetectionPage = () => {
    const [image, setImage] = useState(null);
    const [imageUrl, setImageUrl] = useState(null);
    const [imageloading, setImageloading] = useState(false);
    const [uploadError, setUploadError] = useState(null);
    const [uploadSuccess, setUploadSuccess] = useState(false);
    const [public_id,setPublic_id]=useState("")
    const [detectError, setDetectError] = useState(null);
    const [detectSuccess, setDetectSuccess] = useState(false);
    const [detectloading, setDetectloading] = useState(false);
    const [numberPlate,setNumberPlate]=useState("");
    const [formattedOutput,setFormattedOutput]=useState([]);

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
      setFormattedOutput(arrayofnumberplates)
      console.log(arrayofnumberplates)
    }

    
  
    const handleFileChange = (e) => {
      const file = e.target.files[0];
      setImage(file);
      setImageUrl(URL.createObjectURL(file));
      setNumberPlate("")
      setUploadSuccess(false)
      setDetectSuccess(false)
    };
  
    const handleUpload = () => {
      setImageloading(true);
      setUploadError(null);
      setUploadSuccess(false);
  
      const formData = new FormData();
      formData.append('file', image);
      formData.append('upload_preset', 'prashant-vehicle');
  
      fetch(`https://api.cloudinary.com/v1_1/prashantneupane/image/upload`, {
        method: 'POST',
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          console.log(data.public_id);
          setPublic_id(data.public_id)
          setUploadSuccess(true);
          setImageloading(false);
        })
        .catch((error) => {
          console.error(error);
          setUploadError(error);
          setImageloading(false);
        });
    };

    const handleDetection=()=>
    {
      setDetectloading(true);
      setDetectError(null);
      setDetectSuccess(false);
      var detection_url="http://127.0.0.1:8000/image-detection?public_id="+public_id
      fetch(detection_url, {
        method: 'GET',
      })
        .then((response) => response.json())
        .then((data) => {
          console.log(data.message);
          setNumberPlate(data.message)
          handleOutputFormat(data.message);
          setDetectSuccess(true)
          setDetectloading(false);
        })
        .catch((error) => {
          console.error(error);
          setDetectError(error);
          setDetectloading(false);
        });
    }
  
    return (
      <Box sx={{ mt: 2 }}>
        <Grid container spacing={2}>
        <Grid item xs={12}>
          <Typography variant="h5">Upload and Detect Nepali Number Plates</Typography>
        </Grid>
          <Grid item xs={12} sm={6}>
          <Paper sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <input type="file" name="file" accept="jpeg" onChange={handleFileChange} />
        {imageUrl && (
          <Box mt={2}>
              <img src={imageUrl} alt="Preview" style={{ maxWidth: '400px',maxHeight: '400px' }} />
          </Box>
        )}
        </Box>
            <Button variant="contained" disabled={!image || imageloading} onClick={handleUpload}>
              Upload
            </Button>
            <Box mt={2}>
            <Button variant="contained" disabled={!uploadSuccess || detectloading} onClick={handleDetection}>
              Detect Number Plate
            </Button>
          </Box>
          </Paper>
          </Grid>
          <Grid item xs={12} sm={6}>
          <Paper sx={{ p: 2 }}>
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
          <Grid container sx={{ mt: 2 }} alignItems="right">
            <Grid item xs={12} sm={6}>
              <Typography variant="body1" color="success">
                <List>
                  The Number Plates detected in the image are:
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
            </Grid>
          </Grid>
        )}
        </Paper>
        </Grid>
        </Grid>
      </Box>
    );
  };

export default DetectionPage;
