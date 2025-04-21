import argparse
import asyncio
import logging
import os
import numpy as np
from datetime import datetime
from typing import Literal, List

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
)

logger = logging.getLogger(__name__)


class TemplateForecaster(ForecastBot):
    """
    Enhanced template bot for Q2 2025 Metaculus AI Tournament.
    
    Key enhancements:
    - Increased parallel question processing (6 questions concurrently)
    - Multiple forecasts per question with statistical aggregation
    """

    _max_concurrent_questions = 6  # Increased from 2 to 6 for faster parallel processing
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    
    # Number of forecasts to generate per question
    _forecasts_per_question = 8

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                logger.info(f"Using AskNews for research on question {question.page_url}")
                searcher = AskNewsSearcher()
                research = await searcher.invoke_and_return_formatted_articles(question.question_text)
            elif os.getenv("EXA_API_KEY") and os.getenv("OPENAI_API_KEY"):
                logger.info(f"Using Exa SmartSearcher for research on question {question.page_url}")
                searcher = SmartSearcher(
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                )
                prompt = clean_indents(
                    f"""
                    You are an assistant to a superforecaster. The superforecaster will give
                    you a question they intend to forecast on. To be a great assistant, you generate
                    a concise but detailed rundown of the most relevant news, including if the question
                    would resolve Yes or No based on current information. You do not produce forecasts yourself.
                    
                    The question is: {question.question_text}
                    """
                )
                research = await searcher.invoke(prompt)
            elif os.getenv("OPENROUTER_API_KEY"):
                logger.info(f"Using OpenRouter for research on question {question.page_url}")
                research = await self._call_perplexity(
                    question.question_text, use_open_router=True
                )
            else:
                logger.warning(
                    f"No research provider found when processing question URL {question.page_url}. Will pass back empty string."
                )
                research = ""
            
            logger.info(
                f"Found Research for URL {question.page_url}:\n{research[:500]}..."
            )
            return research

    async def _call_perplexity(
        self, question: str, use_open_router: bool = False
    ) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question}

            Try to find base rates/historical rates or any way that the current situation is different from history
            """
        )
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning"
        else:
            model_name = "perplexity/sonar-pro"
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        response = await model.invoke(prompt)
        return response

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """
        Generate multiple forecasts for binary questions and statistically aggregate them.
        """
        forecasts = []
        reasonings = []
        
        # Generate multiple forecasts
        for i in range(self._forecasts_per_question):
            # Vary temperature slightly to get diverse forecasts
            temperature = 0.2 + (i * 0.05)  # Temperature increases slightly with each forecast
            
            prompt = clean_indents(
                f"""
                You are a professional forecaster interviewing for a job.

                Your interview question is:
                {question.question_text}

                Question background:
                {question.background_info}

                This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
                {question.resolution_criteria}

                {question.fine_print}

                Your research assistant says:
                {research}

                Today is {datetime.now().strftime("%Y-%m-%d")}.

                Before answering you write:
                (a) The time left until the outcome to the question is known.
                (b) The status quo outcome if nothing changed.
                (c) A brief description of a scenario that results in a No outcome.
                (d) A brief description of a scenario that results in a Yes outcome.
                (e) Please consider the historical base rate and make a guess if you're not sure

                You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

                The last thing you write is your final answer as: "Probability: ZZ%", 0-100
                """
            )
            
            reasoning = await self.get_llm("default", "llm").invoke(prompt, temperature=temperature)
            reasonings.append(reasoning)
            
            prediction: float = PredictionExtractor.extract_last_percentage_value(
                reasoning, max_prediction=1, min_prediction=0
            )
            forecasts.append(prediction)
            
            logger.info(
                f"Generated forecast {i+1}/{self._forecasts_per_question} for {question.page_url}: {prediction}"
            )
        
        # Calculate aggregate statistics
        mean_forecast = np.mean(forecasts)
        median_forecast = np.median(forecasts)
        min_forecast = min(forecasts)
        max_forecast = max(forecasts)
        std_dev = np.std(forecasts)
        
        # Use median as the final prediction (more robust to outliers)
        final_prediction = median_forecast
        
        # Create combined reasoning with all individual forecasts and statistics
        combined_reasoning = self._create_combined_binary_reasoning(
            reasonings, forecasts, final_prediction
        )
        
        logger.info(
            f"Aggregated {len(forecasts)} forecasts for {question.page_url}. "
            f"Range: {min_forecast:.3f}-{max_forecast:.3f}, Mean: {mean_forecast:.3f}, "
            f"Median: {median_forecast:.3f}, StdDev: {std_dev:.3f}. Final: {final_prediction:.3f}"
        )
        
        return ReasonedPrediction(
            prediction_value=final_prediction, reasoning=combined_reasoning
        )

    def _create_combined_binary_reasoning(
        self, reasonings: List[str], forecasts: List[float], final_prediction: float
    ) -> str:
        """Create a combined reasoning with statistical information"""
        combined = "## Statistical Analysis of Multiple Forecasts\n\n"
        combined += f"I've generated {len(forecasts)} different forecasts to optimize prediction accuracy.\n\n"
        combined += f"Statistical summary:\n- Range: {min(forecasts):.3f} to {max(forecasts):.3f}\n"
        combined += f"- Mean: {np.mean(forecasts):.3f}\n- Median: {np.median(forecasts):.3f}\n"
        combined += f"- Standard deviation: {np.std(forecasts):.3f}\n\n"
        combined += f"Final prediction (using median): {final_prediction:.3f} ({final_prediction*100:.1f}%)\n\n"
        
        # Include detailed breakdown of forecasts
        combined += "## Individual Forecast Breakdown\n\n"
        for i, (forecast, reasoning) in enumerate(zip(forecasts, reasonings)):
            combined += f"### Forecast {i+1}: {forecast:.3f} ({forecast*100:.1f}%)\n\n"
            combined += reasoning + "\n\n"
        
        return combined

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        """
        Generate multiple forecasts for multiple choice questions and statistically aggregate them.
        """
        all_predictions = []
        all_reasonings = []
        
        # Generate multiple forecasts
        for i in range(self._forecasts_per_question):
            # Vary temperature slightly to get diverse forecasts
            temperature = 0.2 + (i * 0.05)
            
            prompt = clean_indents(
                f"""
                You are a professional forecaster interviewing for a job.

                Your interview question is:
                {question.question_text}

                The options are: {question.options}

                Background:
                {question.background_info}

                {question.resolution_criteria}

                {question.fine_print}

                Your research assistant says:
                {research}

                Today is {datetime.now().strftime("%Y-%m-%d")}.

                Before answering you write:
                (a) The time left until the outcome to the question is known.
                (b) The status quo outcome if nothing changed.
                (c) A description of an scenario that results in an unexpected outcome.

                You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

                The last thing you write is your final probabilities for the N options in this order {question.options} as:
                Option_A: Probability_A
                Option_B: Probability_B
                ...
                Option_N: Probability_N
                """
            )
            
            reasoning = await self.get_llm("default", "llm").invoke(prompt, temperature=temperature)
            all_reasonings.append(reasoning)
            
            prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
            all_predictions.append(prediction)
            
            logger.info(
                f"Generated multiple choice forecast {i+1}/{self._forecasts_per_question} for {question.page_url}"
            )
        
        # Aggregate predictions for each option
        aggregated_prediction = {}
        options = question.options
        
        for option in options:
            option_values = [pred.get(option, 0.0) for pred in all_predictions]
            aggregated_prediction[option] = np.median(option_values)
        
        # Normalize to ensure probabilities sum to 1.0
        total_probability = sum(aggregated_prediction.values())
        for option in aggregated_prediction:
            aggregated_prediction[option] /= total_probability
        
        # Create combined reasoning
        combined_reasoning = self._create_combined_multiple_choice_reasoning(
            all_reasonings, all_predictions, aggregated_prediction
        )
        
        logger.info(
            f"Aggregated {len(all_predictions)} multiple choice forecasts for {question.page_url}"
        )
        
        return ReasonedPrediction(
            prediction_value=aggregated_prediction, reasoning=combined_reasoning
        )

    def _create_combined_multiple_choice_reasoning(
        self, reasonings: List[str], predictions: List[PredictedOptionList], final_prediction: PredictedOptionList
    ) -> str:
        """Create combined reasoning for multiple choice questions"""
        combined = "## Statistical Analysis of Multiple Forecasts\n\n"
        combined += f"I've generated {len(predictions)} different forecasts to optimize prediction accuracy.\n\n"
        
        # Statistical summary for each option
        combined += "### Statistical Summary by Option\n\n"
        
        # Collect all options
        all_options = list(final_prediction.keys())
        
        for option in all_options:
            option_probs = [pred.get(option, 0) for pred in predictions]
            combined += f"**{option}**:\n"
            combined += f"- Range: {min(option_probs):.3f} to {max(option_probs):.3f}\n"
            combined += f"- Mean: {np.mean(option_probs):.3f}\n- Median: {np.median(option_probs):.3f}\n"
            combined += f"- Standard deviation: {np.std(option_probs):.3f}\n"
            combined += f"- Final value (median): {final_prediction[option]:.3f}\n\n"
        
        # Include detailed breakdown of forecasts
        combined += "## Individual Forecast Breakdown\n\n"
        for i, reasoning in enumerate(reasonings):
            combined += f"### Forecast {i+1}:\n\n"
            combined += reasoning + "\n\n"
        
        return combined

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        Generate multiple forecasts for numeric questions and statistically aggregate them.
        """
        all_predictions = []
        all_reasonings = []
        
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        
        # Generate multiple forecasts
        for i in range(self._forecasts_per_question):
            # Vary temperature slightly to get diverse forecasts
            temperature = 0.2 + (i * 0.05)
            
            prompt = clean_indents(
                f"""
                You are a professional forecaster interviewing for a job.

                Your interview question is:
                {question.question_text}

                Background:
                {question.background_info}

                {question.resolution_criteria}

                {question.fine_print}

                Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

                Your research assistant says:
                {research}

                Today is {datetime.now().strftime("%Y-%m-%d")}.

                {lower_bound_message}
                {upper_bound_message}

                Formatting Instructions:
                - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
                - Never use scientific notation.
                - Always start with a smaller number (more negative if negative) and then increase from there

                Before answering you write:
                (a) The time left until the outcome to the question is known.
                (b) The outcome if nothing changed.
                (c) The outcome if the current trend continued.
                (d) The expectations of experts and markets.
                (e) A brief description of an unexpected scenario that results in a low outcome.
                (f) A brief description of an unexpected scenario that results in a high outcome.

                You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

                The last thing you write is your final answer as:
                "
                Percentile 10: XX
                Percentile 20: XX
                Percentile 40: XX
                Percentile 60: XX
                Percentile 80: XX
                Percentile 90: XX
                "
                """
            )
            
            reasoning = await self.get_llm("default", "llm").invoke(prompt, temperature=temperature)
            all_reasonings.append(reasoning)
            
            prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
            all_predictions.append(prediction)
            
            logger.info(
                f"Generated numeric forecast {i+1}/{self._forecasts_per_question} for {question.page_url}"
            )
        
        # Aggregate the predictions - for numeric distributions, we take the median of each percentile
        aggregated_percentiles = {}
        
        # Collect all percentiles used across predictions
        all_percentiles = set()
        for prediction in all_predictions:
            for percentile in prediction.declared_percentiles.keys():
                all_percentiles.add(percentile)
        
        # For each percentile, take the median value across all predictions
        for percentile in sorted(all_percentiles):
            values = []
            for prediction in all_predictions:
                if percentile in prediction.declared_percentiles:
                    values.append(prediction.declared_percentiles[percentile])
            
            if values:
                aggregated_percentiles[percentile] = np.median(values)
        
        # Create a new NumericDistribution with the aggregated percentiles
        template_prediction = all_predictions[0]  # Use the first prediction as a template
        template_prediction.declared_percentiles = aggregated_percentiles
        
        # Create combined reasoning
        combined_reasoning = self._create_combined_numeric_reasoning(
            all_reasonings, all_predictions, template_prediction
        )
        
        logger.info(
            f"Aggregated {len(all_predictions)} numeric forecasts for {question.page_url}"
        )
        
        return ReasonedPrediction(
            prediction_value=template_prediction, reasoning=combined_reasoning
        )

    def _create_combined_numeric_reasoning(
        self, reasonings: List[str], predictions: List[NumericDistribution], final_prediction: NumericDistribution
    ) -> str:
        """Create combined reasoning for numeric questions"""
        combined = "## Statistical Analysis of Multiple Forecasts\n\n"
        combined += f"I've generated {len(predictions)} different forecasts to optimize prediction accuracy and better estimate uncertainty bounds.\n\n"
        
        # Statistical summary for each percentile
        combined += "### Statistical Summary by Percentile\n\n"
        
        # Collect all percentiles
        all_percentiles = sorted(final_prediction.declared_percentiles.keys())
        
        for percentile in all_percentiles:
            percentile_values = [pred.declared_percentiles.get(percentile, None) for pred in predictions]
            percentile_values = [v for v in percentile_values if v is not None]
            
            if percentile_values:
                combined += f"**Percentile {percentile}**:\n"
                combined += f"- Range: {min(percentile_values)} to {max(percentile_values)}\n"
                if len(percentile_values) > 1:
                    combined += f"- Mean: {np.mean(percentile_values)}\n"
                    combined += f"- Median: {np.median(percentile_values)}\n"
                    combined += f"- Standard deviation: {np.std(percentile_values)}\n"
                combined += f"- Final value (median): {final_prediction.declared_percentiles[percentile]}\n\n"
        
        # Include detailed breakdown of forecasts
        combined += "## Individual Forecast Breakdown\n\n"
        for i, reasoning in enumerate(reasonings):
            combined += f"### Forecast {i+1}:\n\n"
            combined += reasoning + "\n\n"
        
        return combined

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Enhanced Metaculus Forecasting Bot"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = (
        args.mode
    )
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # We're generating multiple forecasts in our custom code
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
    )

    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore