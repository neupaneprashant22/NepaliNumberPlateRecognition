import React from 'react';
import { Typography } from  '@mui/material';

const NumberPlateSubHeading = ({ text }) => {
  return (
    <Typography variant="h6" component="h7" sx={{ color: 'white', fontWeight: 'bold' }}>
      An easy to use application to detect nepali vehicle's number plate
    </Typography>
  );
};

export default NumberPlateSubHeading;
