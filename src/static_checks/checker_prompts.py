neg_verification_prompt = """I am going to give you two questions, P and Q, and I want you to validate wether Q is the negation of P.
So if P is true, Q should be false, and if P is false, Q should be true.

Example 1:

P: What is probability of getting a 6 when rolling a dice?
Q: What is probability of not getting a 6 when rolling a dice?

reasoning: Both events are mutually exclusive and exhaustive. So if the probability of getting a 6 is 1/6, the probability of not getting a 6 is 5/6.
valid: True

Example 2:

P: What is the probability of getting a 6 when rolling a dice?
Q: What is the probability of getting a 5 when rolling a dice?

reasoning: These two events are not mutually exclusive. In most cases, both will be false.
valid: False

----

P: {P}
Q: {not_P}
"""

and_verification_prompt = """I am going to give you three questions P, Q and R, and I want you to validate wether R is the conjunction of P and Q.
This means that R is true if and only if both P and Q are true.

Example 1:

P: What is the probability of getting a 6 when rolling a dice?
Q: What is the probability of getting a 5 when rolling a dice?
R: What is the probability of getting a number larger than 4 when rolling a dice?

reasoning: The event of getting a number larger than 4 is the conjunction of getting a 5 and getting a 6.
valid: True

Example 2:
P: What is the probability of Tesla switching to gasoline cars before 2030?
Q: What is the probability of Elon Musk becoming the president of the United States before 2030?
R: What is the probability of Spain leaving the European Union before 2030?

reasoning: None of the events are related to each other. So the probability of Spain leaving the EU is not the conjunction of the other two events.
valid: False

----

P: {P}
Q: {Q}
R: {P_and_Q}
"""

or_verification_prompt = """I am going to give you three questions P, Q and R, and I want you to validate wether R is the disjunction of P and Q.
This means that R is true if at least one of P or Q is true.

Example 1:

P: What is the probability of the republican party winning the 2024 US elections?
Q: What is the probability of the democratic party winning the 2024 US elections?
R: What is the probability of one of two major US parties winning the 2024 US elections? 

reasoning: The event of one of the two major US parties winning the 2024 US elections is the disjunction of the republican party winning and the democratic party winning.
valid: True

Example 2:

P: What is the probability of the republican party winning the 2024 US elections?
Q: What is the probability of the democratic party winning the 2024 US elections?
R: What is the probability of an atheist winning the 2024 US elections?

reasoning: Either P or Q can happen, but R is not related to either of them.
valid: False

----

P: {P}
Q: {Q}
R: {P_or_Q}
"""

but_verification_prompt = """
I will present you with three questions: P, R, and S. In these questions:
- P is a straightforward proposition.
- R is intended to represent another question Q, combined with the negation of P (not P).
- S represents the disjunction of P and Q.

Your task is to validate whether the relationships between P, Q, R, and S are logically consistent and plausible.

Example 1:

P: What is the probability of a Spanish manned mission to the moon before 2030?
R: What is the probability of the US manned mission to the moon before 2030 and there not being a Spanish manned mission before 2030?
S: What is the probability of either a Spanish manned mission to the moon before 2030 or a US manned mission before 2030?

Reasoning: Assuming Q is the occurrence of a US manned mission before 2030, R combines this with the negation of P, i.e., no Spanish mission before 2030. S is true if either mission happens. The definitions for R as Q and not P, and S as P or Q, are logically coherent.
Valid: True

Example 2:

P: What is the probability of a Spanish manned mission to the moon before 2030?
R: What is the probability of both a Spanish and a US manned mission to the moon before 2030?
S: What is the probability of either a Spanish manned mission to the moon before 2030 or a US manned mission before 2030?

Reasoning: In this scenario, R doesnt contain the negation of P, so the relationship between P, Q, R, and S is not the ones we are looking for.
Valid: False

----

P: {P}
R: {Q_and_not_P}
S: {P_or_Q}
"""

conditional_verification_prompt = """
I will provide three questions: P, R, and S. In this scenario:
- P is a direct statement or proposition.
- R represents the conditional probability of Q given P, denoting how likely Q is to occur if P is true.
- S represents the conjunction of P and Q, indicating that both P and Q must be true.

Your task is to validate whether the relationships between P, Q, R, and S are logically consistent and plausible.

Example 1:

P: Will it rain tomorrow?
R: Will there be a thunderstorm tomorrow given that it rains?
S: Will it rain tomorrow and there be a thunderstorm?

Reasoning: Assuming Q is the occurrence of a thunderstorm, R evaluates the probability of a thunderstorm given that it rains (P). S checks if both events, rain and thunderstorm, occur simultaneously. Thus, the conditions for R and S logically follow from the definitions of conditional probability and conjunction respectively.
Valid: True

Example 2:

P: Will it rain tomorrow?
R: If tomorrow is sunny, will it rain?
S: Will it be sunny tomorrow and rain?

Reasoning: In this scenario, R does not represent the conditional probability of Q given P, and S does not represent the conjunction of P and Q. The relationships between P, Q, R, and S are not logically consistent.
Valid: False

----

P: {P}
R: {Q_given_P}
S: {P_and_Q}
"""

consequence_verification_prompt = """
I will provide you with two propositions, P and Q. Your task is to assess whether Q is a proposition that will always be true if P is true. In other words, validate whether Q is a logical implication of P, ensuring that Q will always occur if P is true. Reject if P and Q are completely equivalent. Reject if you need any additional assumptions to derive Q from P.

Example 1:

P: A computer can receive emails.
Q: A computer is connected to the internet.

Reasoning: If a computer can receive emails (P), then it must be connected to the internet (Q), as an internet connection is necessary for receiving emails. Or, contrapositively: If a computer is not connected to the internet, it cannot receive emails. Therefore, Q is a logical consequence of P, meaning that if P is true, Q must also be true.

Valid: True

Example 2:

P: The ground is wet.
Q: It is raining.

Reasoning: If the ground is wet (P), it is possible it could be from rain (Q). However, the ground could also be wet for other reasons (such as sprinklers or a spilled bucket, or rain in the recent past), which means that P does not strictly require Q to be true. Therefore, P can be true without Q necessarily being true.
Valid: False

Example 3:

P: It is daytime.
Q: The sun has risen and not set yet.

Reasoning: The two statements are logically equivalent, as daytime (P) is defined by the sun being above the horizon and not having set yet (Q). So Q is a logical consequence of P; however, they are completely equivalent and therefore not useful to us.
Valid: False

Example 4:

P: Will at least 50 percent of the world's population live in Asia by 2050?
Q: Will Asia have at least 3 billion residents by 2050?

Reasoning: They probably thought Q was a logical consequence of P because the world population is 8 billion, half of that is 4 billion, so if Asia has more than 4 billion people it must have more than 3 billion people. However, this assumes that the world population in 2050 is 8 billion, which we do not know for certain. Without knowing the world population in 2050, we cannot judge if 50 percent of that is more or less than 3 billion.
Valid: False

Example 5:

P: Will ANY of the following happen in 2025? (a) A manned mission to Mars (b) A new Starship launch by SpaceX?
Q: Will a manned mission to Mars happen in 2025?

Reasoning: No, P is a disjunction of two events, either of which happening will make P true. Q is a specific event, which is a subset of the events in P. So P can be true without Q being true.
Valid: False

----

P: {P}
Q: {cons_P}
"""

paraphrase_verification_prompt = """
I will present two questions, Q and P. Your task is to determine if they are paraphrases of each other, meaning they ask the same question but potentially in different ways.

Example 1:

P: What is the current temperature in New York City?
Q: How warm is it right now in New York City?

Reasoning: Both questions inquire about the present temperature in New York City. Despite slight differences in phrasing ("current temperature" vs. "how warm"), the essential inquiry is the same, focusing on the temperature at this moment.
Valid: True

Example 2:

P: Who won the Best Actor award at the Oscars last year?
Q: Who was the recipient of the last Academy Award for Best Actress?

Reasoning: These questions appear similar as they both pertain to Oscar winners, but they ask about different categories—Best Actor and Best Actress. Therefore, they are not paraphrases because they request distinct information.
Valid: False

Example 3:

P: Can dogs eat chocolate?
Q: Is it safe for dogs to consume chocolate?

Reasoning: Both questions address the safety of chocolate consumption for dogs. The essence of the inquiry is whether chocolate poses a risk to dogs, which makes them paraphrases of each other despite the subtle difference in wording.
Valid: True

Example 4:

P: How far is it from Earth to Mars?
Q: What is the distance between Earth and the Moon?

Reasoning: Although both questions ask about distances in space, they refer to entirely different celestial measurements. Hence, they are not paraphrases as they inquire about different distances.
Valid: False

----

P: {P}
Q: {para_P}
"""
