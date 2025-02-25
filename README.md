## Emotion Yoga Chatbot

### Overview
This project is a chatbot that classifies emotions using BERT and provides personalized yoga recommendations based on detected emotions. It utilizes deep learning and NLP techniques to analyze user input and suggest yoga poses to promote mental well-being.

### Features
- Emotion detection using a fine-tuned BERT model.
- Derived emotion categories for better mental health analysis.
- Personalized yoga pose recommendations based on emotional state.
- Interactive chatbot for user engagement.

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/emotion-yoga-chatbot.git
   cd emotion-yoga-chatbot
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Usage
Run the chatbot:
```sh
python chatbot.py
```
Follow the prompts to describe your emotions and receive tailored yoga pose recommendations.

### Dependencies
Add the following packages to `requirements.txt`:
```
pandas
scikit-learn
tensorflow
transformers
```

### Contributing
Feel free to submit issues and pull requests to improve the chatbot!

### License
This project is licensed under the MIT License.
