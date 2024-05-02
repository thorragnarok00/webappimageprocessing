import React, { useState } from 'react';
import { View, Text, Image, Button, StyleSheet } from 'react-native';
import axios from 'axios';

const App = () => {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleImageChange = async (event) => {
    const file = event.target.files[0];
    const imageURL = URL.createObjectURL(file);
    
    // Fetch the image data from the URL
    try {
      const response = await fetch(imageURL);
      const blob = await response.blob();
      setImage(blob); // Set the image data (blob) as the state
    } catch (error) {
      console.error('Error fetching image:', error);
    }
  };
  
  const predictImage = async () => {
    if (!image) {
      console.error('No image data provided.');
      return;
    }

    const formData = new FormData();
    formData.append('image', image);
  
    try {
      console.log('Sending prediction request...');
      console.log('Form Data:', formData); // Log FormData to verify image is appended correctly
      const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('Prediction response:', response.data);
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('Prediction failed:', error);
    }
  };
  
  return (
    <View style={styles.container}>
      <View>
        <Text style={styles.headerText}>Apple or Orange?</Text>
      </View>
      <View style={styles.imageContainer}>
        {image && <Image source={{ uri: URL.createObjectURL(image) }} style={styles.image} />}
      </View>
      <View style={styles.buttonContainer}>
        <input type="file" accept="image/*" onChange={handleImageChange} style={styles.fileInput} />
        <Button title="Predict" onPress={predictImage} disabled={!image} />
      </View>
      <View style={styles.predictionContainer}>
        {prediction && <Text style={styles.predictionText}><Text>Prediction:</Text> <Text style={styles.boldText}>{prediction}</Text></Text>}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f0f0f0',
    minHeight: 500,
  },
  headerText: {
    fontSize: 36,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  imageContainer: {
    marginBottom: 20,
  },
  image: {
    width: 300,
    height: 300,
    resizeMode: 'cover',
  },
  buttonContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  fileInput: {
    marginRight: 10,
  },
  predictionContainer: {
    height: 40,
  },
  predictionText: {
    fontSize: 20,
  },
  boldText: {
    fontWeight: 'bold',
  },
});

export default App;
