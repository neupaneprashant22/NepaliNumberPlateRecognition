import React from 'react';
import { Typography } from  '@mui/material';

const NumberPlateHeading = ({ text }) => {
  return (
    <Typography variant="h3" component="h1" sx={{ color: 'white', fontWeight: 'bold' }}>
      Nepali Number Plate Detection System
    </Typography>
  );
};

export default NumberPlateHeading;
