import React, { useState } from 'react';
import { Cloudinary } from 'cloudinary-react';

const CloudinaryUploader = () => {
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setImageUrl(URL.createObjectURL(file));
  };

  const handleUpload = () => {
    setLoading(true);
    setUploadError(null);
    setUploadSuccess(false);

    const formData = new FormData();
    formData.append('file', image);
    formData.append('upload_preset', 'prashant-vehicle'); // replace with your Cloudinary upload preset

    fetch(`https://api.cloudinary.com/v1_1/prashantneupane/image/upload`, {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log(data.public_id);
        setUploadSuccess(true);
        setLoading(false);
      })
      .catch((error) => {
        console.error(error);
        setUploadError(error);
        setLoading(false);
      });
  };

  return (
    <div>
      <input type="file" name="file" onChange={handleFileChange} />
      {imageUrl && <img src={imageUrl} alt="Preview" />}
      <button onClick={handleUpload}>Upload</button>

      {loading && <p>Uploading image...</p>}
      {uploadError && <p>Upload error: {uploadError}</p>}
      {uploadSuccess && <p>Upload successful!</p>}
    </div>
  );
};

export default CloudinaryUploader;
