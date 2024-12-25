from HRBot import HRBot
from deepeval.red_teaming import RedTeamer
from dotenv import load_dotenv
import os
from deepeval.red_teaming import AttackEnhancement
from deepeval.vulnerability import PIILeakage, PromptLeakage, IntellectualProperty
from deepeval.vulnerability.pii_leakage import PIILeakageType
from deepeval.vulnerability.prompt_leakage import PromptLeakageType
from deepeval.vulnerability.intellectual_property import IntellectualPropertyType


load_dotenv()

target_llm = HRBot()

target_purpose = "Provide HR policies, and answer user queries related to company"
target_system_prompt = target_llm.get_system_prompt()

red_teamer = RedTeamer(
    target_purpose=target_purpose,
    target_system_prompt=target_system_prompt,
    synthesizer_model=os.getenv("INFERENCE_MODEL", 'gpt-3.5-turbo-0125"'),
    evaluation_model=os.getenv("INFERENCE_MODEL", "gpt-3.5-turbo"),
    async_mode=False
)

sensitive_information_vulnerabilities = [
    PIILeakage(types=[
        PIILeakageType.DATABASE_ACCESS,
        PIILeakageType.DIRECT,
        PIILeakageType.SESSION_LEAK,
        PIILeakageType.SOCIAL_MANIPULATION
    ]),
    PromptLeakage(types=[
        PromptLeakageType.SECRETS_AND_CREDENTIALS,
        PromptLeakageType.INSTRUCTIONS,
        PromptLeakageType.GUARDS,
        PromptLeakageType.PERMISSIONS_AND_ROLES
    ]),
    IntellectualProperty(types=[
        IntellectualPropertyType.PATENT_DISCLOSURE
    ])
]

results = red_teamer.scan(
    target_model_callback=target_llm,
    attacks_per_vulnerability_type=5,
    vulnerabilities=sensitive_information_vulnerabilities,
    attack_enhancements={
        AttackEnhancement.BASE64: 0.25,
        AttackEnhancement.GRAY_BOX_ATTACK: 0.25,
        AttackEnhancement.JAILBREAK_CRESCENDO: 0.25,
        AttackEnhancement.MULTILINGUAL: 0.25,
    },
)
print("Red Teaming Results: ", results)