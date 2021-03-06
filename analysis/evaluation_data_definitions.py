

import pandas as pd

from attrs import define, field
from cattrs import structure, unstructure

list_field = field(factory=list)
dict_field = field(factory=dict)


@define
class WorkUnit:
    id: str = None
    worker_id: str = None
    time_to_complete_sec: float | None = None
    task: str = None
    labels: list[str] = list_field
    dialogue_ids: list[str] = list_field


@define
class LikertAnnotation:
    work_unit_id: str = None
    label: str = None
    score: int = None


@define
class BehaviorAnnotation:
    work_unit_id: str = None
    label: str = None
    score: int = None


@define
class ComparativeAnnotation:
    work_unit_id: str = None
    label: str = None
    dialogue_compared_to_id: str = None
    score: int = None
    bot: str = None
    bot_compared_to: str = None


@define
class TrainingResult:
    work_unit_id: str = None
    mistakes: list[int] | None = list_field
    performance: float | None = None
    passed: bool | None = None


@define
class TurnPair:
    user_turn: str = None
    bot_turn: str = None
    likert_annotations: dict[str, list[LikertAnnotation]] = dict_field
    behavior_annotations: dict[str, list[BehaviorAnnotation]] = dict_field


@define
class Dialogue:
    dialogue_id: str = None
    bot: str = None
    comparative_annotations: dict[str, list[ComparativeAnnotation]] = dict_field
    likert_annotations: dict[str, list[LikertAnnotation]] = dict_field
    turns: list[TurnPair] = list_field

    def all_likert_dialogue_annotations(self):
        for label, annotations in self.likert_annotations.items():
            for annotation in annotations:
                yield self.dialogue_id, annotation

    def all_comparative_annotations(self):
        for label, annotations in self.comparative_annotations.items():
            for annotation in annotations:
                yield self.dialogue_id, annotation

    def all_likert_turn_annotations(self):
        for i, turn in enumerate(self.turns):
            turn_id = (self.dialogue_id, i)
            for label, annotations in turn.likert_annotations.items():
                for annotation in annotations:
                    yield turn_id, annotation

    def all_behavior_annotations(self):
        for i, turn in enumerate(self.turns):
            turn_id = (self.dialogue_id, i)
            for label, annotations in turn.behavior_annotations.items():
                for annotation in annotations:
                    yield turn_id, annotation

    def annotations(self):
        return {
            'likert dialogue': list(self.all_likert_dialogue_annotations()),
            'comparative': list(self.all_comparative_annotations()),
            'likert turn': list(self.all_likert_turn_annotations()),
            'behavior': list(self.all_behavior_annotations())
        }


@define
class OnboardingDialogue(Dialogue):
    attempts: list[TrainingResult] = list_field


@define
class Evaluation:
    dialogues: dict[str, Dialogue] = dict_field
    work_units: dict[str, WorkUnit] = dict_field

    def likert_dialogue_annotations(self):
        for _, dialogue in self.dialogues.items():
            yield from dialogue.all_likert_dialogue_annotations()

    def comparative_annotations(self):
        for _, dialogue in self.dialogues.items():
            yield from dialogue.all_comparative_annotations()

    def likert_turn_annotations(self):
        for _, dialogue in self.dialogues.items():
            yield from dialogue.all_likert_turn_annotations()

    def behavior_annotations(self):
        for _, dialogue in self.dialogues.items():
            yield from dialogue.all_behavior_annotations()

    def annotations(self):
        return {
            'likert dialogue': list(self.likert_dialogue_annotations()),
            'comparative': list(self.comparative_annotations()),
            'likert turn': list(self.likert_turn_annotations()),
            'behavior': list(self.behavior_annotations())
        }

    def annotation_dataframe(self):
        marks = {}
        for label_category, annotations in self.annotations().items():
            for item, annotation in annotations:
                did, tid = item if isinstance(item, tuple) else (item, None)
                bot = self.dialogues[did].bot
                marks.setdefault(
                    (bot, label_category, annotation.label, item), []
                ).append(annotation.score)
        df = pd.DataFrame(marks.values(), marks)
        df.index.set_names(['bot', 'category', 'label', 'item'], inplace=True)
        return df

    def comparative_annotation_dataframe(self):
        marks = {}
        for item, annotation in self.comparative_annotations():
            bot1 = annotation.bot
            bot2 = annotation.bot_compared_to
            item = (item, annotation.dialogue_compared_to_id)
            marks.setdefault(
                (bot1, bot2, annotation.label, item), []
            ).append(annotation.score)
        df = pd.DataFrame(marks.values(), marks)
        df.index.set_names(['bot', 'bot comp', 'label', 'dialogues'], inplace=True)
        return df

    def annotation_counts(self):
        """
        :return: Dataframe of number of dialogues annotated per category label
        """
        # {(category, label): list of annotated dialogues}
        annotated = {}
        # {(category, label): {'dialogues annotated': int, 'double annotated': int}}
        annotated_counts = {}
        for category, annotations in self.annotations().items():
            for item, annotation in annotations:
                did, tid = item if isinstance(item, tuple) else (item, 0)
                if not tid:
                    annotated.setdefault(
                        (category, annotation.label), []
                    ).append(did)
        for (category, label), annotations in annotated.items():
            unique_annotated = set(annotations)
            num_annotated = len(unique_annotated)
            for ua in unique_annotated:
                annotations.remove(ua)
            num_double_annotated = len(set(annotations))
            annotated_counts[(category, label)] = {
                'dialogues annotated': num_annotated,
                'double annotated': num_double_annotated
            }
        return pd.DataFrame(annotated_counts.values(), annotated_counts)

@define
class OnboardingEvaluation(Evaluation):
    dialogues: dict[str, OnboardingDialogue] = dict_field
    work_units: dict[str, WorkUnit] = dict_field


@define
class Project:
    annotation_pilots: list[Evaluation] = list_field
    annotation_pilots_onboarding: list[OnboardingEvaluation] = list_field
    bot_pilots: list[Evaluation] = list_field
    extra_unused: Evaluation | None = None
    dialogue_collection: Evaluation | None = None
    student_evaluation: Evaluation | None = None
    student_onboarding: OnboardingEvaluation | None = None
    student_gold_units: Evaluation | None = None
    mturk_evaluation: Evaluation | None = None
    mturk_onboarding: OnboardingEvaluation | None = None
    mturk_gold_units: Evaluation | None = None
    surge_evaluation: Evaluation | None = None
    surge_onboarding: OnboardingEvaluation | None = None
    surge_gold_units: Evaluation | None = None
    expert_evaluation: Evaluation | None = None



__all__ = [
    "WorkUnit",
    "LikertAnnotation",
    "BehaviorAnnotation",
    "ComparativeAnnotation",
    "TrainingResult",
    "TurnPair",
    "Dialogue",
    "OnboardingDialogue",
    "Evaluation",
    "OnboardingEvaluation",
    "Project"
]

if __name__ == '__main__':

    t = Evaluation()
    print(unstructure(t))
