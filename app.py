from flask import Flask, render_template, request
import joblib
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load model & vectorizer
model = joblib.load("Trained Model/fake_job_model.pkl")
vectorizer = joblib.load("Trained Model/tfidif_vectorizer.pkl")

app = Flask(__name__)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def extract_text_from_image(image_file):
    """
    Extract text from uploaded image using OCR
    """
    try:
        # Read the image file
        image = Image.open(image_file)
        
        # Convert to RGB if necessary (for JPEG compatibility)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using pytesseract
        extracted_text = pytesseract.image_to_string(image)
        
        return extracted_text.strip()
    
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    job_desc = ""

    if request.method == "POST":
        action = request.form.get("action")

        if action == "check":
            job_desc = request.form.get("job_desc", "").strip()
            
            print(f"Debug - Text input: '{job_desc}'")
            print(f"Debug - Files in request: {list(request.files.keys())}")
            
            # Check if an image was uploaded
            if 'job_image' in request.files:
                image_file = request.files['job_image']
                print(f"Debug - Image file: {image_file}")
                print(f"Debug - Image filename: '{image_file.filename}'")
                
                if image_file and image_file.filename != '' and image_file.filename is not None:
                    print("Debug - Processing uploaded image...")
                    # Extract text from image using OCR
                    try:
                        extracted_text = extract_text_from_image(image_file)
                        print(f"Debug - Extracted text: '{extracted_text[:100]}...'")
                        
                        if extracted_text:
                            # If we have both text input and image, combine them
                            if job_desc:
                                job_desc = f"{job_desc}\n\n{extracted_text}"
                                print("Debug - Combined text input and OCR text")
                            else:
                                job_desc = extracted_text
                                print("Debug - Using only OCR text")
                        else:
                            if not job_desc:
                                prediction = "⚠️ No text could be extracted from the image. Please try a clearer image or enter text manually."
                                return render_template("index.html", prediction=prediction, job_desc="")
                    except Exception as e:
                        print(f"Debug - OCR Error: {e}")
                        if not job_desc:
                            prediction = f"❌ Error processing image: {str(e)}"
                            return render_template("index.html", prediction=prediction, job_desc="")
                else:
                    print("Debug - No valid image file found")
            
            # Perform prediction if we have any text
            if job_desc:
                try:
                    print(f"Debug - Running prediction on text (length: {len(job_desc)})")
                    job_desc_vectorized = vectorizer.transform([job_desc])
                    pred = model.predict(job_desc_vectorized)[0]
                    prediction = "🚨 Fake Job" if pred == 1 else "✅ Real Job"
                    print(f"Debug - Prediction result: {prediction}")
                except Exception as e:
                    prediction = f"❌ Error analyzing job posting: {str(e)}"
                    print(f"Debug - Prediction error: {e}")
            else:
                prediction = "⚠️ Please provide job description text or upload an image with readable text"
                print("Debug - No text available for prediction")

        elif action == "clear":
            job_desc = ""
            prediction = None

    return render_template("index.html", prediction=prediction, job_desc=job_desc)

if __name__ == "__main__":
    app.run(debug=True)

