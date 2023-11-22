file_name = "345.txt"

emotion_counts = {"그리움": 0, "사랑 기쁨": 0, "설렘 심쿵": 0, "스트레스 짜증": 0, "외로움": 0, "슬픔": 0}

with open(file_name, 'r', encoding='utf-8') as file:
    for line in file:
        for emotion in emotion_counts:
            if emotion in line:
                emotion_counts[emotion] += 1

total_count = sum(emotion_counts.values())
emotion_percentages = {emotion: (count / total_count) * 100 for emotion, count in emotion_counts.items()}

for emotion, percentage in emotion_percentages.items():
    print(f"{emotion}: {percentage:.2f}%")