import uuid

from vital_llm_reasoner.config.reasoner_config import ReasonerConfig
from vital_llm_reasoner.ensemble.member.answer import Answer
from vital_llm_reasoner.ensemble.member.ensemble_member import EnsembleMember
from vital_llm_reasoner.ensemble.member.inquiry import Inquiry
from vital_llm_reasoner.ensemble.member.task_tag import TaskTag
import asyncio
from pyppeteer import launch
from black import format_str, FileMode


class CodeExecutorMember(EnsembleMember):

    @classmethod
    def get_task_tag(cls) -> TaskTag:
        task_tag = TaskTag('code_executor')
        return task_tag

    def handle_inquiry(self, inquiry: Inquiry) -> Answer:

        code_string = inquiry.inquiry

        # should be regexes
        code_string = "\n".join(line for line in code_string.splitlines() if "```python" not in line)

        code_string = "\n".join(line for line in code_string.splitlines() if "```" not in line)

        # hack for encoded tabs?
        # code_string = "\n".join(
        #    line.replace("\\t", "\t")  # Replace \t with an actual tab
        #    for line in code_string.splitlines()
        #)

        answer_dict = asyncio.run(self.run_python_with_pyodide(code_string))

        random_guid = uuid.uuid4()

        answer_string = str(answer_dict) + f"\nCode Execution Confirmation: {random_guid}.\n"

        answer = Answer(inquiry=inquiry, answer=answer_string)

        return answer

    async def run_python_with_pyodide(self, code_string):

        formatted_code = None

        try:
            formatted_code = format_str(code_string, mode=FileMode())
        except Exception as e:
            error = f"{type(e).__name__}: {e}\nBe sure your indentation is correct."
            return error

        code_string = formatted_code

        # Launch the browser in headless mode
        browser = await launch(headless=True)
        page = await browser.newPage()

        # Load Pyodide via CDN
        await page.goto(
            'data:text/html,<script src="https://cdn.jsdelivr.net/pyodide/v0.23.0/full/pyodide.js"></script>')

        # Evaluate Python code in Pyodide, capturing stdout
        result = await page.evaluate(f"""
            async () => {{
                const pyodide = await loadPyodide();
                try {{
                    let output = "";
                    const captureOutput = (text) => {{ output += text + "\\n"; }};

                    // Redirect stdout in Pyodide
                    pyodide.runPython(`
    import sys
    from io import StringIO
    sys.stdout = StringIO()
                    `);

                    // Run the provided code
                    pyodide.runPython(`{code_string}`);

                    // Retrieve stdout
                    const std_output = pyodide.runPython("sys.stdout.getvalue()");
                    return {{ success: true, output: std_output }};
                }} catch (error) {{
                    return {{ success: false, error: `${{error.name}}: ${{error.message}}` }};
                }}
            }}
        """)

        # Close the browser instance
        await browser.close()
        return result
