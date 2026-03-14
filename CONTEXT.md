## Background 
 Fraud detection in financial transactions has traditionally relied on rule-based systems 
to flag suspicious activities. These systems, however, often struggle with evolving fraud 
patterns, leading to high false-positive rates and increased manual review. To improve 
adaptability and reduce false positives, many organizations now incorporate machine 
learning models that can learn from historical transaction data. This assignment 
provides transaction data, case histories, and rule-based criteria, challenging you to 
develop a model that utilizes machine learning or/and rule-based insights to enhance 
fraud detection accuracy and interpretability. 
 
## Context of Problem 
 In a debit card transaction, several key participants interact: the Merchant (the 
business or seller), the Acquirer (the bank that processes transactions on behalf of the 
merchant), and the Client (the debit cardholder who initiates the transaction). As 
the card switch provider, our role is to ensure secure transaction processing and 
monitor for potential fraud. 
• Real-Time Authorization Decisioning (RAD): Pre-authorization, rule-based 
fraud check in real-time to flag suspicious transactions before approval. 
• Case Creation (CC): Post-authorization, this process flags transactions that 
initially passed RAD checks but still display potential fraud indicators, prompting 
investigation by a Fraud Analyst. 
Note: RAD and CC may follow diJerent sets of rules, depending on acquirer 
configurations. 
When transactions are flagged, the Fraud Analyst may contact the client to verify 
legitimacy and close cases upon confirmation (see “CASE_STATUS” and 
“F_TYPE_CODE” in Case_Creation_Details table). Together, RAD and CC provide a 
layered fraud prevention strategy, combining immediate transaction screening with in-
depth post-event analysis. 
 
## Objective 
Develop a fraud detection model that can identify suspicious transactions based on 
historical data and potentially integrate predefined rules. This assignment assesses 
your skills in data processing, feature engineering, model development, and 
interpretability. 
 
## Data Available 
 The following anonymized datasets pertain to one of our clients (referred to as "The 
Acquirer" or XYZ). Each dataset includes a "Field" tab in its table, which provides 
descriptions and map_values (where applicable): 
 
1. ISO8583_Transaction_Records.pdf - Issuer transaction data (ISO 8583), detailing 
transactions processed by the issuer.  
2. Dataset/Case_Creation_Details.xlsx - Details of cases created based on flagged 
transactions. 
3. Dataset/Case_Update_History.xlsx - Historical updates on cases in the Case Creation (CC) 
process. 
4. Dataset/Main_Rules_Table.xlsx - Primary rules for identifying suspicious activities. 
5. Dataset/Rules_Details.xlsx - Specifics about each rule. 
6. Dataset/Transaction_Hit_Rules.xlsx - Data on transactions that triggered RAD/CC rules.

Notes: For DE### related columns, refer to ISO8583_Spec_21May2024.pdf attached 
for the interpretation. 

## Relationship
graph TD
    %% Define the container
    subgraph Conceptual_Schema [Conceptual Schema]
        direction TB

        %% Top Row
        CUH[Case_Update_History] --- CCD[Case_Creation_Details]

        %% Middle Row
        ISO["⭐ ISO8583_Transaction_Records"] ---|CASE_NO| THR[Transaction_Hit_Rules]

        %% Bottom Row
        MRT[Main_Rules_Table] --- RD[Rule_Details]

        %% Vertical Connections
        THR ---|CASE_NO| CUH
        THR ---|RULE_ID| MRT

    end