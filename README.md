<p align="center">
    <h1 align="center">
        <img src="assets/genie_worksheets_circle.png" width=100px>
        <br>
        <b>GenieWorksheets</b>
        <br>
        <a href="https://arxiv.org/abs/2407.05674">
            <img src="https://img.shields.io/badge/cs.CL-2407.05674-b31b1b"
            alt="arXiv">
        </a>
        <a href="https://ws.genie.stanford.edu/">
            <img src="https://img.shields.io/badge/website-genie.stanford.edu-blue"
            alt="Website">
        </a>
        <a href="https://ws.genie.stanford.edu/docs/">
            <img src="https://img.shields.io/badge/docs-genie.stanford.edu-blue"
            alt="Docs">
        </a>
    </h1>
</p>
<p align="center">
    Framework for creating reliable conversational agents
</p>


Genie is a programmable framework for creating task-oriented conversational
agents that are designed to handle complex user interactions and knowledge
access. Unlike LLMs, Genie provides reliable grounded responses, with 
controllable agent policies through its expressive specification, Genie 
Worksheet. In contrast to dialog trees, it is resilient to diverse user queries,
helpful with knowledge sources, and offers ease of programming policies through
 its declarative paradigm.

[Research Preprint](https://arxiv.org/abs/2407.05674): To be presented at ACL 2025

<img src="assets/banner.jpg">

## Installation

To install Genie, we recommend using uv ([UV installation guide](https://github.com/astral-sh/uv?tab=readme-ov-file#installation))


```bash
git clone https://github.com/stanford-oval/genie-worksheets.git
cd worksheets
uv venv
source venv/bin/activate
uv sync
```

## Creating Agents

Example agents are present in `experiments/agents/` directory. You can use them
as a reference to create your own agents.

### Load the model configuration

```python
from worksheets import Config
import os

# Define path to the prompts
current_dir = os.path.dirname(os.path.realpath(__file__))
prompt_dir = os.path.join(current_dir, "prompts")

# Load config from YAML file
config = Config.load_from_yaml(os.path.join(current_dir, "config.yaml"))
```

You can also define the configuration programmatically:

```python
from worksheets import Config, AzureModelConfig
import os

config = Config(
    semantic_parser=AzureModelConfig(
        model_name="azure/gpt-4o",
    ),
    response_generator=AzureModelConfig(
        model_name="azure/gpt-4o",
    ),
    knowledge_parser=AzureModelConfig(
        model_name="azure/gpt-4o",
    ),
    knowledge_base=AzureModelConfig(
        model_name="azure/gpt-4o",
    ),
    prompt_dir=prompt_dir,
)
```

### Define your API functions

```python
from worksheets.agent.config import agent_api
from worksheets.core.worksheet import get_genie_fields_from_ws
from uuid import uuid4

@agent_api("course_detail_to_individual_params", "Get course details")
def course_detail_to_individual_params(course_detail):
    if course_detail.value is None:
        return {}
    course_detail = course_detail.value
    course_detail = {}
    for field in get_genie_fields_from_ws(course_detail):
        course_detail[field.name] = field.value

    return course_detail

@agent_api("courses_to_take_oval", "Final API to enroll into a course")
def courses_to_take_oval(**kwargs):
    return {"success": True, "transaction_id": uuid4()}

@agent_api("is_course_full", "Check if a course is full")
def is_course_full(course_id, **kwargs):
    # Implementation here
    return False
```

### Define your starting prompt

You can load your starting prompt from a template file:

```python
from worksheets.agent.builder import TemplateLoader

starting_prompt = TemplateLoader.load(
    os.path.join(current_dir, "starting_prompt.md"), format="jinja2"
)
```

Or define it inline:

```python
starting_prompt = """Hello! I'm the Course Enrollment Assistant. I can help you with:
- Selecting a course: just say find me programming courses
- Enrolling into a course. 
- Asking me any question related to courses and their requirement criteria.

How can I help you today?"""
```

### Define the Agent

```python
from worksheets import AgentBuilder, SUQLKnowledgeBase, SUQLReActParser

agent = (
    AgentBuilder(
        name="Course Enrollment Assistant",
        description="You are a course enrollment assistant. You can help students with course selection and enrollment.",
        starting_prompt=starting_prompt.render() if hasattr(starting_prompt, 'render') else starting_prompt,
    )
    .with_knowledge_base(
        SUQLKnowledgeBase,
        tables_with_primary_keys={
            "courses": "course_id",
            "ratings": "rating_id",
            "offerings": "course_id",
            "programs": "program_id",
        },
        database_name="course_assistant",
        embedding_server_address="http://127.0.0.1:8509",
        source_file_mapping={
            "course_assistant_general_info.txt": os.path.join(
                current_dir, "course_assistant_general_info.txt"
            )
        },
        postprocessing_fn=postprocess_suql,
        result_postprocessing_fn=None,
        db_username="select_user",
        db_password="select_user",
    )
    .with_parser(
        SUQLReActParser,
        example_path=os.path.join(current_dir, "examples.txt"),
        instruction_path=os.path.join(current_dir, "instructions.txt"),
        table_schema_path=os.path.join(current_dir, "table_schema.txt"),
    )
    .with_gsheet_specification("YOUR_SPREADSHEET_ID_HERE")
    .build(config)
)
```

### Load the agent from a CSV file
You can also load the agent from a CSV file. Instead of using the `with_gsheet_specification` method, you can use the `with_csv_specification` method.

```python
from worksheets import AgentBuilder, SUQLKnowledgeBase, SUQLReActParser
import os

agent = (
    AgentBuilder(
        name="Course Enrollment Assistant",
        description="You are a course enrollment assistant. You can help students with course selection and enrollment.",
        starting_prompt=starting_prompt.render() if hasattr(starting_prompt, 'render') else starting_prompt,
    )
    .with_csv_specification(os.path.join(current_dir, "course_enrollment.csv"))
    .build(config)
)
```

### Load the agent from a JSON file
You can also load the agent from a JSON file. Instead of using the `with_gsheet_specification` method, you can use the `with_json_specification` method.

```python
from worksheets import AgentBuilder, SUQLKnowledgeBase, SUQLReActParser
import os

agent = (
    AgentBuilder(
        name="Course Enrollment Assistant",
        description="You are a course enrollment assistant. You can help students with course selection and enrollment.",
        starting_prompt=starting_prompt.render() if hasattr(starting_prompt, 'render') else starting_prompt,
    )
    .with_json_specification(os.path.join(current_dir, "course_enrollment.json"))
    .build(config)
)
```

A sample JSON file is present in `experiments/domain_agents/course_enroll/course_enrollment.json`.

### Run the conversation loop

```python
import asyncio
from worksheets import conversation_loop

if __name__ == "__main__":
    # Run the conversation loop in the terminal
    asyncio.run(conversation_loop(agent, "output_state_path.json", debug=True))
```


### Add prompts
For each agent you need to create prompts for:
- Semantic parsing: `semantic_parsing.prompt`
- Response generation: `response_generator.prompt`

Place these prompts in the prompt directory that you specify while creating the
agent.

You can copy basic annotated prompts from `experiments/sample_prompts/` 
directory. Make change where we have `TODO`. You need two provide a few 
guidelines in the prompt that will help the LLM to perform better and some 
examples. Please `experiments/domain_agents/course_enroll/prompts/` for inspiration!


### Spreadsheet Specification

To create a new agent, you should have a Google Service Account and create a new spreadsheet. 
You can follow the instructions [here](https://cloud.google.com/iam/docs/service-account-overview) to create a Google Service Account.
Share the created spreadsheet with the service account email.

You should save the service_account key as `service_account.json` in the `worksheets/` directory.

Here is a starter worksheet that you can use for your reference: [Starter Worksheet](https://docs.google.com/spreadsheets/d/1ST1ixBogjEEzEhMeb-kVyf-JxGRMjtlRR6z4G2sjyb4/edit?usp=sharing)

Here is a sample spreadsheet for a restaurant agent: [Restaurant Agent](https://docs.google.com/spreadsheets/d/1FXg5VFrdxQlUyld3QmKKL9BN1lLIhAtQTJjCHyNOU_Y/edit?usp=sharing)

Please note that we only use the specification defined in the first sheet of the spreadsheet.

## LLM Config
You should create a `.env` file similar to `.env.example` and fill in the values for the LLM API keys and endpoints.

### Running the Agent (Web Interface)

Create a folder `frontend/`  under `experiments/agents/<agent_name>` and create a `app_*` file.

You can run the agent in a web interface by running:

**NOTE:** You should run the agent in the `frontend` directory to preserve the frontend assets.

For restaurant agent:
```bash
cd experiments/domain_agents/yelpbot/frontend/
chainlit run app_restaurant.py --port 8800
```

## Cite our work

If you use Genie in your research or applications, please cite our work:

```
@article{genieworksheets,
  title={Coding Reliable LLM-based Integrated Task and Knowledge Agents with GenieWorksheets},
  author={Joshi, Harshit and Liu, Shicheng and Chen, James and Weigle, Robert and Lam, Monica S},
  journal={arXiv preprint arXiv:2407.05674},
  year={2024}
}
```

GenieWorksheets logo is designed with the help of DALL-E.