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
I will provide you with two propositions, P and Q. Your task is to assess whether Q is a proposition that could only be true if P is true. In other words, validate whether Q is logically contingent on P, ensuring that Q can only occur if P is true.

Example 1:

P: A computer is connected to the internet.
Q: A computer can receive emails.

reasoning: If a computer is connected to the internet (P), then it is capable of receiving emails (Q). If P is false (the computer is not connected to the internet), then Q must also be false (the computer cannot receive emails), since receiving emails inherently requires an internet connection. Thus, Q's truth is contingent upon P being true.
valid: True

Example 2:

P: It is raining.
Q: The ground is wet.

reasoning: If it is raining (P), then the ground is likely to be wet (Q). However, the ground could also be wet for other reasons (such as sprinklers or a spilled bucket), which means that Q does not strictly require P to be true. Therefore, Q can be true without P necessarily being true.
valid: False

Example 3:

P: The car engine is running.
Q: The car is moving.

reasoning: If the car engine is running (P), it is possible for the car to be moving (Q), but the car could be stationary (e.g., if it's in neutral or parked with the handbrake applied). Hence, Q can be true or false independently of P. Q does not exclusively depend on P being true.
valid: False

----

P: {P}
Q: {cons_P}
"""

paraphrase_verification_prompt = """
I will present two questions, Q and P. Your task is to determine if they are paraphrases of each other, meaning they ask the same question but in different ways.

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

reasoning: Although both questions ask about distances in space, they refer to entirely different celestial measurements. Hence, they are not paraphrases as they inquire about different distances.
valid: False

Example 5:

P: Will more than 100 million people in the world use GLP-1 agonists daily in 2032?
Q: Is it likely that more than 100 million people worldwide will be using GLP-1 agonists daily in 2032?

reasoning: The second question is asking about whether it should be considered probable that more than 100 million people will use GLP-1 agonists daily in 2032, while the first question asks for a definitive answer on whether this will happen.
valid: False

----

P: {P}
Q: {para_P}
"""



