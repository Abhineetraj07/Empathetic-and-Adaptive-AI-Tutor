from gamified_review import launch_gamified_review


def evaluate_and_adapt(score, topic_name, video_description=""):
    if score >= 6:
        print(f"\n Learner understood '{topic_name}' well — moving to next lecture.")
        return "next_topic"
    else:
        print(f"\n Learner struggled with '{topic_name}' — launching gamified review!")
        points = launch_gamified_review(
            topic_name=topic_name,
            quiz_csv="quiz_results.csv",
            emotion_csv="emotion_data.csv",
            video_description=video_description
        )
        print(f"\n Review complete! Student earned {points} points.")
        return "reteach"
