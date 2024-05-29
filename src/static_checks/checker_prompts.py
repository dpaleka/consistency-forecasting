neg_verification_prompt = """I am going to give you two questions, P and Q, and I want you to validate whether Q is the negation of P.
So if P is true, Q should be false, and if P is false, Q should be true.

Example 1:

P:
  Title: Will the Second Amendment of the United States Constitution be amended or repealed before 2025?
  Body: This question will resolve as Yes if, at any time between January 1, 2019 to January 1, 2025, the Second Amendment is repealed, amended, or modified in any way through the procedure in Article Five of the US Constitution.
        
Q:
  Title: Will the Second Amendment of the United States Constitution remain unchanged until 2025?
  Body: This question will resolve as Yes if, from January 1, 2019 to January 1, 2025, the Second Amendment is not repealed, amended, or modified in any way through the procedure in Article Five of the US Constitution.
        
reasoning: Both events are mutually exclusive and exhaustive. So if the Second Amendment is amended or repealed before 2025, it cannot remain unchanged until 2025, and vice versa.
valid: True

Example 2:

P:
  Title: Will any of Belgium, France, Italy, Luxembourg, Netherlands, and/or Germany leave the EU before 2027?
  Body: This question will resolve as Yes if, before January 1, 2027, one or more of Belgium, France, Italy, Luxembourg, Netherlands, and/or Germany triggers Article 50 of the Treaty of Lisbon by notifying the European Council of its intention to leave the union.
        It also resolves as positive if by the same deadline the entity called European Union has dissolved completely or dissolved to create a new politico-economic union of some European nations under a different name.

Q:
  Title: Will none of Belgium, France, Italy, Luxembourg, Netherlands, and/or Germany leave the EU before 2027?
  Body: This question will resolve as Yes if, before January 1, 2027, none of Belgium, France, Italy, Luxembourg, Netherlands, and/or Germany triggers Article 50 of the Treaty of Lisbon by notifying the European Council of its intention to leave the union.
        It also resolves as positive if by the same deadline the entity called European Union remains unchanged.
        
reasoning: These two events are not mutually exclusive. It is possible for both to be false if the EU dissolves to create a similar international union under a different name.
valid: False

----

P title: {P_title}
P body: {P_body}
Q title: {not_P_title}
Q body: {not_P_body}
"""

and_verification_prompt = """I am going to give you three questions P, Q, and R, and I want you to validate whether R is the logical conjunction of P and Q.
This means that R is true if and only if both P and Q are true.

Example 1:

P:
  Title: What is the probability of getting a 6 when rolling a dice?
  Body: The event occurs if a 6 appears on the top face when a fair six-sided die is rolled.
Q:
  Title: What is the probability of getting a 5 when rolling a dice?
  Body: The event occurs if a 5 appears on the top face when a fair six-sided die is rolled.
R:
  Title: What is the probability of getting a number larger than 4 when rolling a dice?
  Body: The event occurs if either a 5 or a 6 appears on the top face when a fair six-sided die is rolled.

reasoning: The event of getting a number larger than 4 is the conjunction of getting a 5 and getting a 6.
valid: True

Example 2:

P:
  Title: What is the probability of Tesla switching to gasoline cars before 2030?
  Body: The event occurs if Tesla announces and implements a complete switch to gasoline-powered cars before January 1, 2030.
Q:
  Title: What is the probability of Elon Musk becoming the president of the United States before 2030?
  Body: The event occurs if Elon Musk is inaugurated as the President of the United States at any time before January 1, 2030.
R:
  Title: What is the probability of Spain leaving the European Union before 2030?
  Body: The event occurs if Spain formally triggers Article 50 of the Treaty on European Union to leave the EU before January 1, 2030.

reasoning: The event of Spain leaving the EU has little to do with the other two events, so it is definitely not their conjunction.
valid: False

----

P title: {P_title}
P body: {P_body}
Q title: {Q_title}
Q body: {Q_body}
R title: {R_title}
R body: {R_body}
"""

or_verification_prompt = """I am going to give you three questions P, Q, and R, and I want you to validate whether R is the logical disjunction of P and Q.
This means that R is true if at least one of P or Q is true.

Example 1:

P:
  Title: Will the Republican Party win the 2024 US elections?
  Body: The event occurs if the Republican candidate is elected as the President of the United States in the 2024 elections.
Q:
  Title: Will the Democratic Party win the 2024 US elections?
  Body: The event occurs if the Democratic candidate is elected as the President of the United States in the 2024 elections.
R:
  Title: What is the probability of one of the two major US parties winning the 2024 US elections?
  Body: The event occurs if either the Republican candidate or the Democratic candidate is elected as the President of the United States in the 2024 elections.

reasoning: The event of one of the two major US parties winning the 2024 US elections is the disjunction of the Republican Party winning and the Democratic Party winning.
valid: True

Example 2:

P:
  Title: What is the probability of the republican party winning the 2024 US elections?
  Body: The event occurs if the Republican party candidate is elected as the President of the United States in the 2024 elections.
Q:
  Title: What is the probability of the democratic party winning the 2024 US elections?
  Body: The event occurs if the Democratic party candidate is elected as the President of the United States in the 2024 elections.
R:
  Title: What is the probability of an atheist winning the 2024 US elections?
  Body: The event occurs if a candidate who identifies as an atheist is elected as the President of the United States in the 2024 elections.

reasoning: Either P or Q can happen, but R is not related to either of them.
valid: False

----

P title: {P_title}
P body: {P_body}
Q title: {Q_title}
Q body: {Q_body}
R title: {R_title}
R body: {R_body}
"""

but_verification_prompt = """
I will present you with three questions: P, R, and S. In these questions:
- P is a question (no specific requirements).
- R is intended to represent another question Q, combined with the negation of P (not P).
- S represents the logical disjunction of P and Q.

Your task is to validate whether the relationships between P, Q, R, and S are logically consistent and plausible.

Example 1:

P:
  Title: What is the probability of a Spanish manned mission to the moon before 2030?
  Body: The event occurs if a Spanish crewed spacecraft lands on the moon before January 1, 2030.
R:
  Title: What is the probability of a US manned mission to the moon before 2030 and there not being a Spanish manned mission before 2030?
  Body: The event occurs if a US crewed spacecraft lands on the moon before January 1, 2030, and a Spanish crewed spacecraft does not land on the moon before January 1, 2030.
S:
  Title: What is the probability of either a Spanish manned mission to the moon before 2030 or a US manned mission before 2030?
  Body: The event occurs if either a Spanish crewed spacecraft or a US crewed spacecraft lands on the moon before January 1, 2030.

reasoning: Assuming Q is the occurrence of a US manned mission before 2030, R combines this with the negation of P, i.e., no Spanish mission before 2030. S is true if either mission happens. The definitions for R as Q and not P, and S as P or Q, are logically coherent.
valid: True

Example 2:

P:
  Title: What is the probability of a Spanish manned mission to the moon before 2030?
  Body: The event occurs if a Spanish crewed spacecraft lands on the moon before January 1, 2030.
R:
  Title: What is the probability of both a Spanish and a US manned mission to the moon before 2030?
  Body: The event occurs if both a Spanish crewed spacecraft and a US crewed spacecraft land on the moon before January 1, 2030.
S:
  Title: What is the probability of either a Spanish manned mission to the moon before 2030 or a US manned mission before 2030?
  Body: The event occurs if either a Spanish crewed spacecraft or a US crewed spacecraft lands on the moon before January 1, 2030.

reasoning: In this scenario, R doesn't contain the negation of P, so the relationship between P, Q, R, and S is not the ones we are looking for.
valid: False

----

P title: {P_title}
P body: {P_body}
R title: {R_title}
R body: {R_body}
S title: {S_title}
S body: {S_body}
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

reasoning: Assuming Q is the occurrence of a thunderstorm, R evaluates the probability of a thunderstorm given that it rains (P). S checks if both events, rain and thunderstorm, occur simultaneously. Thus, the conditions for R and S logically follow from the definitions of conditional probability and conjunction respectively.
valid: True

Example 2:

P: Will it rain tomorrow?
R: If tomorrow is sunny, will it rain?
S: Will it be sunny tomorrow and rain?

reasoning: In this scenario, R does not represent the conditional probability of Q given P, and S does not represent the conjunction of P and Q. The relationships between P, Q, R, and S are not logically consistent.
valid: False

----

P: {P}
R: {Q_given_P}
S: {P_and_Q}
"""

consequence_verification_prompt = """
I will provide you with two propositions, P and Q. Your task is to assess whether Q is a proposition that will always be true if P is true. In other words, validate whether Q is a logical implication of P, ensuring that Q will always occur if P is true. Reject if P and Q are completely equivalent. Reject if you need any additional assumptions to derive Q from P. Reject if Q is just formed by making some resolution criteria more vague / not operationalizing them (but accept if it is made by actually loosening some resolution criteria while still precisely defining everything). Reject if Q is 'ERROR: NO CONSEQUENCE FOUND' or something like that.

Example 1:

P: A computer can receive emails.
Q: A computer is connected to the internet.

reasoning: If a computer can receive emails (P), then it must be connected to the internet (Q), as an internet connection is necessary for receiving emails. Therefore, Q is a logical consequence of P.
valid: True

Example 2:

P: The ground is wet.
Q: It is raining.

reasoning: I can easily imagine the ground being wet (P true) without it raining (Q false). So P does not imply Q.
valid: False

Example 3:

P: It is daytime.
Q: The sun has risen and not set yet.

reasoning: The two statements are logically equivalent, as daytime (P) is defined by the sun being above the horizon and not having set yet (Q). So Q is a logical consequence of P, but also completely equivalent to it, therefore not useful to us.
valid: False

Example 4:

P: Will at least 50 percent of the world's population live in Asia by 2050?
Q: Will Asia have at least 3 billion residents by 2050?

reasoning: They probably thought Q was a logical consequence of P because the world population is 8 billion, half of that is 4 billion, so if Asia has more than 4 billion people it must have more than 3 billion people. However, this assumes that the world population in 2050 is 8 billion, which we do not know for certain. Without knowing the world population in 2050, we cannot judge if 50 percent of that is more or less than 3 billion.
valid: False

Example 5:

P: Will ANY of the following happen in 2025? (a) A manned mission to Mars (b) A new Starship launch by SpaceX?
Q: Will a manned mission to Mars happen in 2025?

reasoning: Suppose only a new starship launch happens, but no manned mission to Mars. Then P is true, but Q is false. So Q is not a logical consequence of P.
valid: False

Example 6:

P: Will there be an epidemic of meningococcal diseases in 2025? (An epidemic is defined as a disease with an infection rate of over 15 per 100,000 people over a span of two consecutive weeks.)
Q: Will there be an epidemic of meningococcal diseases in 2025? (An epidemic is defined as a disease that infects a large number of people over a short span of time.)

reasoning: Q just makes the definition of an epidemic more vague, so we do not count it.
valid: False

Example 7:

P: Will there be an epidemic of meningococcal diseases in 2025? (An epidemic is defined as a disease with an infection rate of over 15 per 100,000 people over a span of two consecutive weeks.)
Q: Will there be an epidemic of meningococcal diseases in 2025? (An epidemic is defined as a disease with an infection rate of over 10 per 100,000 people over a span of two consecutive weeks.)

reasoning: Q is a logical consequence of P, as the definition of an epidemic in Q is a subset of the definition of an epidemic in P. If an epidemic is defined as infecting more than 15 per 100,000 people, it must also be true that it infects more than 10 per 100,000 people, as 15 is more than 10.
valid: True

----

P: {P}
Q: {cons_P}
"""

paraphrase_verification_prompt = """
I will present two questions, Q and P. Your task is to determine if they are paraphrases of each other, meaning they ask the same question but potentially in different ways. They should mean the same things, and always have the same answer.

Example 1:

P: What is the current temperature in New York City?
Q: How warm is it right now in New York City?

reasoning: Both questions inquire about the present temperature in New York City. Despite slight differences in phrasing ("current temperature" vs. "how warm"), the essential inquiry is the same, focusing on the temperature at this moment.
valid: True

Example 2:

P: Who won the Best Actor award at the Oscars last year?
Q: Who was the recipient of the last Academy Award for Best Actress?

reasoning: These questions appear similar as they both pertain to Oscar winners, but they ask about different categories—Best Actor and Best Actress. Therefore, they are not paraphrases because they request distinct information.
valid: False

Example 3:

P: Can dogs eat chocolate?
Q: Is it safe for dogs to consume chocolate?

reasoning: Both questions address the safety of chocolate consumption for dogs. The essence of the inquiry is whether chocolate poses a risk to dogs, which makes them paraphrases of each other despite the subtle difference in wording.
valid: True

Example 4:

P: How far is it from Earth to Mars?
Q: What is the distance between Earth and the Moon?

Reasoning: Although both questions ask about distances in space, they refer to entirely different celestial measurements. Hence, they are not paraphrases as they inquire about different distances.
Valid: False

Example 5:

P: Will there be a manned Mars mission in 2025?
Q: Is it probable that a manned mission to Mars will happen in 2025?

reasoning: It's a bit subtle, but these questions do not necessarily have the same answer. For example, it is possible that a manned Mars mission will happen in 2025, but it was not a priori probable; or that a manned Mars mission will not happen in 2025, but it was regarded as probable before that. So they are not paraphrases.
valid: False

----

P: {P}
Q: {para_P}
"""
