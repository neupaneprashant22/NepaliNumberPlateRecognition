import React from 'react';
import { Box, Typography, Button, Grid } from '@mui/material';
import videoBg from './../assets/traffic1.mov'
import NumberPlateHeading from '../Components/NumberPlateHeading';
import NumberPlateSubHeading from '../Components/NumberPlateSubHeading';
import { useNavigate } from "react-router-dom";
const LandingPage = () => {

  const navigate = useNavigate();

  function handleImageClick() {
    navigate("/detection");
  }
  function handleVideoClick() {
    navigate("/videodetection");
  }
  return (
    <Box sx={{ position: 'relative' }}>
      <video src={videoBg} autoPlay muted loop style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'cover', zIndex: -1 }}>
      </video>
      <Box sx={{ position: 'relative' }}>
        <Grid container justifyContent="center" alignItems="center" direction="column" sx={{ height: '100vh' }}>
          <Typography align="center">
            <NumberPlateHeading/>
          </Typography>
          <Typography align="center">
            <NumberPlateSubHeading/>
          </Typography>
          <center>
          <Grid container spacing={2}>
            <Grid item>
              <Button variant="contained" color="primary" onClick={handleImageClick}>
                Image Detection
              </Button>
            </Grid>
            <Grid item>
              <Button variant="contained" color="primary" onClick={handleVideoClick}>
                Video Detection
              </Button>
            </Grid>
          </Grid>
          </center>
        </Grid>
      </Box>
    </Box>
  );
};

export default LandingPage;
