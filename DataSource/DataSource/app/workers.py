from PySide6.QtCore import QThread, Signal

from .services import DatasetGenerationService, TopicExpansionService


class TopicExpansionWorker(QThread):
    progress = Signal(tuple)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, main_topic, target_topic_count, similarity_threshold, batch_size, generation_model_name, validation_model_name, client):
        super().__init__()
        self.main_topic = main_topic
        self.target_topic_count = target_topic_count
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.service = TopicExpansionService(client, generation_model_name, validation_model_name, self.progress)

    def run(self):
        try:
            topics = self.service.generate_topics(
                self.main_topic,
                self.target_topic_count,
                self.similarity_threshold,
                self.batch_size,
            )
            self.finished.emit(topics)
        except Exception as exc:
            self.error.emit(f"A critical error occurred during topic expansion: {exc}")


class DataGenerationWorker(QThread):
    progress = Signal(tuple)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, sub_topics, total_samples, generation_model_name, validation_model_name, client):
        super().__init__()
        self.sub_topics = sub_topics
        self.total_samples = total_samples
        self.service = DatasetGenerationService(client, generation_model_name, validation_model_name, self.progress)

    def run(self):
        try:
            dataset = self.service.generate_dataset(self.sub_topics, self.total_samples)
            self.finished.emit(dataset)
        except Exception as exc:
            self.error.emit(str(exc))
