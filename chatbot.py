import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder


data = pd.read_pickle('/content/merged_training.pkl')  
data = data.head(10000)  


basic_labels = ['Sadness', 'Anger', 'Love', 'Surprise', 'Fear', 'Joy']


label_encoder = LabelEncoder()
data['emotion_encoded'] = label_encoder.fit_transform(data['emotions'])  


X = data['text']
y = data['emotion_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(basic_labels))


train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors='tf')
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors='tf')


train_labels = tf.convert_to_tensor(y_train, dtype=tf.int32)
test_labels = tf.convert_to_tensor(y_test, dtype=tf.int32)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(train_encodings['input_ids'], train_labels,
          validation_data=(test_encodings['input_ids'], test_labels),
          epochs=5,
          batch_size=8)


loss, accuracy = model.evaluate(test_encodings['input_ids'], test_labels)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


def map_to_derived_category(emotion):
    """Maps basic emotions to derived or combined categories."""
    derived_mapping = {
        'Sadness': 'Depression',
        'Anger': 'Stress',
        'Fear': 'Anxiety',
        'Surprise': 'Panic',
        'Sadness + Anger': 'Depression + Stress',
        'Sadness + Fear': 'Depression + Anxiety',
        'Sadness + Disgust': 'Burnout',
        'Fear + Anger': 'Stress + Anxiety',
        'Fear + Surprise': 'Panic + Anxiety',
        'Fear + Disgust': 'Burnout + Anxiety',
        'Anger + Sadness': 'Stress + Burnout',
        'Fear + Sadness + Disgust': 'Complex Burnout',
        'Sadness + Anger (Depression + Burnout)': 'Severe Burnout',
    }
    return derived_mapping.get(emotion, 'Neutral')


def get_yoga_recommendations(derived_category):
    """Provides yoga recommendations based on derived emotional categories."""
    yoga_mapping = {
        'Depression': [
            "Child's Pose (Balasana): A calming and restorative pose.",
            "Seated Forward Fold (Paschimottanasana): Encourages introspection.",
            "Bridge Pose (Setu Bandhasana): Opens the heart and releases sadness."
        ],
        'Stress': [
            "Eagle Pose (Garudasana): Releases tension in shoulders and mind.",
            "Seated Forward Bend (Paschimottanasana): Relieves stress.",
        ],
        'Anxiety': [
            "Mountain Pose (Tadasana): Grounds the body and calms the mind.",
            "Cat-Cow Pose (Marjaryasana-Bitilasana): Alleviates mental tension."
        ],
        'Panic': [
            "Warrior II (Virabhadrasana II): Builds strength and confidence.",
            "Legs-Up-The-Wall Pose (Viparita Karani): Relieves panic and stress."
        ],
        'Burnout': [
            "Reclining Bound Angle Pose (Supta Baddha Konasana): Restores energy.",
            "Child's Pose (Balasana): Promotes deep relaxation."
        ],
        'Neutral': [
            "Tree Pose (Vrikshasana): Promotes balance and positivity.",
            "Savasana (Corpse Pose): Encourages complete relaxation."
        ]
    }
    return yoga_mapping.get(derived_category, ["No specific recommendation available."])


def chat_with_user():
    print("Welcome to your personalized Yoga Journey! 🌸")
    print("I'm here to guide you towards peace and balance. Let's find some yoga poses to help you feel your best.")

    emotion_mapping = {
        "stressed": "Anger",
        "anxious": "Fear",
        "depressed": "Sadness",
        "sad": "Sadness",
        "happy": "Joy",
        "surprised": "Surprise",
        "angry": "Anger",
        "love": "Love",
        "burnout": "Burnout",
        "panic": "Panic"
    }

    while True:
        user_input = input("\nHow are you feeling today? (Type 'exit' to quit): ").lower()

        if user_input == 'exit':
            print("\nThank you for chatting! Wishing you peace and positivity. Take care! ✨")
            break

        detected_emotion = None
        for key in emotion_mapping:
            if key in user_input:
                detected_emotion = emotion_mapping[key]
                break

        if not detected_emotion:
            print("Hmm, I didn't quite catch that. Could you share how you're feeling again? 😊")
            continue

        derived_category = map_to_derived_category(detected_emotion)
        yoga_recommendations = get_yoga_recommendations(derived_category)

        print("\nLet's do some yoga together! 🌿 Here are some poses that may help:")
        for rec in yoga_recommendations:
            print(f"- {rec}")

        print("\nRemember, every breath you take is a step towards calmness. Keep going, you are doing great! 🌟")

if __name__ == "__main__":
    chat_with_user()
