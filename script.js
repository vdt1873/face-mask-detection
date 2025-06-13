
function startDetection() {
  const video = document.getElementById('video');
  video.style.display = 'block';

  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      video.srcObject = stream;

      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');

      setInterval(() => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0);

        canvas.toBlob(blob => {
          const formData = new FormData();
          formData.append('image', blob, 'frame.jpg');

          fetch('https://face-mask-detection-eu0v.onrender.com/detect', {
            method: 'POST',
            body: formData
          })
          .then(response => response.json())
          .then(data => {
            document.getElementById('status').innerText = data.mask ? "Mask" : "No Mask";
            document.getElementById('accuracy').innerText = data.accuracy.toFixed(2) + '%';
          })
          .catch(error => console.error('Error:', error));
        }, 'image/jpeg');
      }, 2000);
    })
    .catch(error => console.error('Camera error:', error));
}
