# Critical Thinking Answers for Community Complaint Categorization

These answers address the three critical-thinking questions in Part 6 of `GUIDE_OG.docx`. They reflect the project's current data fields and implementation.

## 1. Which input features should have the strongest influence on the assigned category?

The complaint text should have the strongest influence because it describes the problem itself. Words and phrases about potholes, blocked drains, leaking pipes, garbage, or broken streetlights provide direct evidence for the five complaint categories. The current classifier converts that text to TF-IDF features and also uses urgency level as a secondary input. Urgency can help distinguish some cases, but it should not outweigh a clear description of the issue: a high-priority water leak is still a water-supply complaint.

The other fields serve different purposes. Location selects the service branch *after* the category is predicted. Urgency also selects the response queue and resolution interval. The date anchors the estimated resolution date, and submission channel records how the complaint arrived. None of those fields should determine the type of complaint by itself; using location or channel to infer category could also produce unfair patterns. `complaint_id` is a reference identifier and should never influence a prediction.

## 2. What should the system do when the available evidence is incomplete or contradictory?

It should ask for missing or clearer information before making a routing decision. The current app rejects empty complaint text, descriptions shorter than three words, invalid field values, and invalid or future dates. If a valid description points to several categories, such as “Flooding damaged the road,” the app shows the predicted category but marks the routing suggestion **Manual review** when the top probability is below 30% or the two leading probabilities differ by less than 10 percentage points. A staff member should then read the complaint and choose or confirm the office.

Those probability checks cannot catch every conflict. For example, a description of an exposed electrical wire paired with a *Low* urgency selection deserves human review even if the category prediction is confident. Staff should also check urgent safety reports promptly rather than rely on an automatic route. The system should preserve the original complaint and record any human correction so the decision can be checked later.

## 3. What could happen if the system produces an incorrect result?

A wrong category can send a complaint to the wrong office or branch. Staff may need to transfer it, delaying service; a missed urgent case may cause property damage or physical harm. Incorrect urgency or location can also put the case in the wrong response queue, while an incorrect resolution estimate can mislead residents about when help will arrive. Repeated errors can reduce trust and, if they affect some areas or submission channels more often, make service unequal.

The current model's test accuracy is 55% for Logistic Regression on a 20-record test set, so its predictions should be treated as suggestions rather than final decisions. Manual review of uncertain and safety-related cases, correction by staff, and evaluation on more representative complaints would reduce the impact of mistakes.
