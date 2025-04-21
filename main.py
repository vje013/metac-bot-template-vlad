import argparse
import asyncio
import logging
import os
import numpy as np
from datetime import datetime
from typing import Literal, List, Dict, Any, Tuple, Optional, Union

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
    - Domain-specific optimizations:
      - Status quo weighting for binary questions
      - Uncertainty bounds optimization for numeric questions
      - Specialized handling for multiple choice questions
    """

    _max_concurrent_questions = 6  # Increased from 2 to 6 for faster parallel processing
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    
    # Number of forecasts to generate per question
    _forecasts_per_question = 8
    
    # Status quo weighting parameter for binary questions (higher = more weight)
    _status_quo_weight = 1.5
    
    # Uncertainty bounds parameter for numeric questions (higher = wider bounds)
    _uncertainty_factor = 1.2

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
                    
                    Please include any relevant historical base rates or statistics to help with forecasting.
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
            
            # Add domain-specific research enhancement
            research = await self._enhance_research_by_question_type(question, research)
            
            logger.info(
                f"Found Research for URL {question.page_url}:\n{research[:500]}..."
            )
            return research
    
    async def _enhance_research_by_question_type(self, question: MetaculusQuestion, base_research: str) -> str:
        """Enhance research based on question type with domain-specific information"""
        if isinstance(question, BinaryQuestion):
            return await self._enhance_binary_research(question, base_research)
        elif isinstance(question, NumericQuestion):
            return await self._enhance_numeric_research(question, base_research)
        elif isinstance(question, MultipleChoiceQuestion):
            return await self._enhance_multiple_choice_research(question, base_research)
        else:
            return base_research
    
    async def _enhance_binary_research(self, question: BinaryQuestion, base_research: str) -> str:
        """Add specialized research for binary questions focusing on status quo and base rates"""
        prompt = clean_indents(
            f"""
            You are a research assistant for a forecaster working on a binary (Yes/No) question.
            
            QUESTION: {question.question_text}
            
            BASE RESEARCH: {base_research}
            
            Please analyze this research and enhance it by addressing:
            1. What is the current status quo related to this question?
            2. What are historical base rates for similar events/questions?
            3. What are the most important factors that could change the status quo?
            
            Format your response as a brief continuation of the original research.
            """
        )
        
        enhancement = await self.get_llm("default", "llm").invoke(prompt)
        return f"{base_research}\n\n### ENHANCED BINARY RESEARCH ###\n{enhancement}"
    
    async def _enhance_numeric_research(self, question: NumericQuestion, base_research: str) -> str:
        """Add specialized research for numeric questions focusing on uncertainty bounds"""
        prompt = clean_indents(
            f"""
            You are a research assistant for a forecaster working on a numeric question.
            
            QUESTION: {question.question_text}
            
            BASE RESEARCH: {base_research}
            
            Please analyze this research and enhance it by addressing:
            1. What are reasonable upper and lower bounds for this question based on historical data?
            2. What are the key sources of uncertainty for this prediction?
            3. Are there statistical patterns or distributions that might be relevant?
            
            Format your response as a brief continuation of the original research.
            """
        )
        
        enhancement = await self.get_llm("default", "llm").invoke(prompt)
        return f"{base_research}\n\n### ENHANCED NUMERIC RESEARCH ###\n{enhancement}"
    
    async def _enhance_multiple_choice_research(self, question: MultipleChoiceQuestion, base_research: str) -> str:
        """Add specialized research for multiple choice questions"""
        prompt = clean_indents(
            f"""
            You are a research assistant for a forecaster working on a multiple choice question.
            
            QUESTION: {question.question_text}
            OPTIONS: {question.options}
            
            BASE RESEARCH: {base_research}
            
            Please analyze this research and enhance it by addressing:
            1. What is the status quo option among the choices?
            2. Are there any historical patterns that favor certain options?
            3. What factors might cause a shift from the status quo to each alternative?
            
            Format your response as a brief continuation of the original research.
            """
        )
        
        enhancement = await self.get_llm("default", "llm").invoke(prompt)
        return f"{base_research}\n\n### ENHANCED MULTIPLE CHOICE RESEARCH ###\n{enhancement}"

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
        Generate multiple forecasts for binary questions with status quo weighting.
        """
        forecasts = []
        reasonings = []
        
        # Extract status quo information from research if possible
        status_quo_indicator = await self._extract_status_quo(question, research)
        
        # Generate multiple forecasts
        for i in range(self._forecasts_per_question):
            # Vary temperature slightly to get diverse forecasts
            temperature = 0.2 + (i * 0.05)  # Temperature increases slightly with each forecast
            
            # For half the forecasts, emphasize status quo more strongly
            status_quo_emphasis = "You write your rationale remembering that good forecasters put EXTRA weight on the status quo outcome since the world changes slowly most of the time." if i < self._forecasts_per_question // 2 else "You write your rationale balancing the status quo with potential change factors."
            
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

                {status_quo_emphasis}

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
        
        # Apply status quo weighting
        weighted_forecasts = self._apply_status_quo_weighting(forecasts, status_quo_indicator)
        
        # Calculate aggregate statistics
        mean_forecast = np.mean(weighted_forecasts)
        median_forecast = np.median(weighted_forecasts)
        min_forecast = min(weighted_forecasts)
        max_forecast = max(weighted_forecasts)
        std_dev = np.std(weighted_forecasts)
        
        # Use median as the final prediction (more robust to outliers)
        final_prediction = median_forecast
        
        # Create combined reasoning with all individual forecasts and statistics
        combined_reasoning = self._create_combined_binary_reasoning(
            reasonings, forecasts, weighted_forecasts, final_prediction, status_quo_indicator
        )
        
        logger.info(
            f"Aggregated {len(forecasts)} forecasts for {question.page_url}. "
            f"Raw range: {min(forecasts):.3f}-{max(forecasts):.3f}, Weighted range: {min_forecast:.3f}-{max_forecast:.3f}, "
            f"Final: {final_prediction:.3f}"
        )
        
        return ReasonedPrediction(
            prediction_value=final_prediction, reasoning=combined_reasoning
        )
    
    async def _extract_status_quo(self, question: BinaryQuestion, research: str) -> Optional[float]:
        """Extract status quo indicator from research"""
        prompt = clean_indents(
            f"""
            Based on the following research for a yes/no question, determine what the status quo is.
            
            Research:
            {research}
            
            Question:
            {question.question_text}
            
            If the status quo appears to be "Yes" (meaning if nothing changes, the outcome would be Yes), respond with "1.0".
            If the status quo appears to be "No" (meaning if nothing changes, the outcome would be No), respond with "0.0".
            If the status quo is unclear or balanced, respond with "0.5".
            
            Respond only with one of these three numbers: 0.0, 0.5, or 1.0
            """
        )
        
        try:
            response = await self.get_llm("default", "llm").invoke(prompt)
            response = response.strip()
            
            if "0.0" in response or "0" in response:
                return 0.0
            elif "1.0" in response or "1" in response:
                return 1.0
            else:
                return 0.5
        except Exception as e:
            logger.warning(f"Error extracting status quo: {e}")
            return None
    
    def _apply_status_quo_weighting(self, forecasts: List[float], status_quo_indicator: Optional[float]) -> List[float]:
        """Apply status quo weighting to forecasts"""
        if status_quo_indicator is None:
            # No information about status quo, return original forecasts
            return forecasts
        
        weighted_forecasts = []
        for forecast in forecasts:
            # Calculate weighted forecast by pulling it toward the status quo based on weight
            weight_factor = 1 - np.exp(-self._status_quo_weight)  # Transform weight to a 0-1 scale
            weighted_forecast = forecast + weight_factor * (status_quo_indicator - forecast)
            weighted_forecasts.append(weighted_forecast)
        
        return weighted_forecasts

    def _create_combined_binary_reasoning(
        self, 
        reasonings: List[str], 
        original_forecasts: List[float], 
        weighted_forecasts: List[float], 
        final_prediction: float,
        status_quo_indicator: Optional[float]
    ) -> str:
        """Create a combined reasoning with statistical information and status quo weighting"""
        status_quo_msg = ""
        if status_quo_indicator is not None:
            if status_quo_indicator == 0.0:
                status_quo_msg = "Analysis indicates the status quo outcome is 'No'."
            elif status_quo_indicator == 1.0:
                status_quo_msg = "Analysis indicates the status quo outcome is 'Yes'."
            else:
                status_quo_msg = "Analysis indicates the status quo is balanced between 'Yes' and 'No'."
        
        combined = "## Statistical Analysis of Multiple Forecasts with Status Quo Weighting\n\n"
        combined += f"I've generated {len(original_forecasts)} different forecasts to optimize prediction accuracy.\n\n"
        
        if status_quo_msg:
            combined += f"**Status Quo Analysis**: {status_quo_msg}\n\n"
        
        combined += f"**Statistical summary (original forecasts)**:\n"
        combined += f"- Range: {min(original_forecasts):.3f} to {max(original_forecasts):.3f}\n"
        combined += f"- Mean: {np.mean(original_forecasts):.3f}\n- Median: {np.median(original_forecasts):.3f}\n"
        combined += f"- Standard deviation: {np.std(original_forecasts):.3f}\n\n"
        
        combined += f"**Statistical summary (with status quo weighting)**:\n"
        combined += f"- Range: {min(weighted_forecasts):.3f} to {max(weighted_forecasts):.3f}\n"
        combined += f"- Mean: {np.mean(weighted_forecasts):.3f}\n- Median: {np.median(weighted_forecasts):.3f}\n"
        combined += f"- Standard deviation: {np.std(weighted_forecasts):.3f}\n\n"
        
        combined += f"**Final prediction (using weighted median)**: {final_prediction:.3f} ({final_prediction*100:.1f}%)\n\n"
        
        # Include detailed breakdown of forecasts
        combined += "## Individual Forecast Breakdown\n\n"
        for i, (original, weighted, reasoning) in enumerate(zip(original_forecasts, weighted_forecasts, reasonings)):
            combined += f"### Forecast {i+1}:\n"
            combined += f"- Original: {original:.3f} ({original*100:.1f}%)\n"
            combined += f"- After status quo weighting: {weighted:.3f} ({weighted*100:.1f}%)\n\n"
            combined += reasoning + "\n\n"
        
        return combined

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        """
        Generate multiple forecasts for multiple choice questions with domain-specific optimizations.
        """
        all_predictions = []
        all_reasonings = []
        
        # Try to identify status quo option
        status_quo_option = await self._identify_status_quo_option(question, research)
        
        # Generate multiple forecasts
        for i in range(self._forecasts_per_question):
            # Vary temperature slightly to get diverse forecasts
            temperature = 0.2 + (i * 0.05)
            
            # For some forecasts, emphasize status quo more strongly
            if status_quo_option and i < self._forecasts_per_question // 2:
                status_quo_emphasis = f"You write your rationale remembering that (1) good forecasters put EXTRA weight on the status quo outcome '{status_quo_option}' since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes."
            else:
                status_quo_emphasis = "You write your rationale remembering that (1) good forecasters put appropriate weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes."
            
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

                {status_quo_emphasis}

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
        
        # Apply domain-specific optimizations
        optimized_prediction = self._optimize_multiple_choice_forecasts(all_predictions, status_quo_option)
        
        # Create combined reasoning
        combined_reasoning = self._create_combined_multiple_choice_reasoning(
            all_reasonings, all_predictions, optimized_prediction, status_quo_option
        )
        
        logger.info(
            f"Aggregated {len(all_predictions)} multiple choice forecasts for {question.page_url}"
        )
        
        return ReasonedPrediction(
            prediction_value=optimized_prediction, reasoning=combined_reasoning
        )
    
    async def _identify_status_quo_option(self, question: MultipleChoiceQuestion, research: str) -> Optional[str]:
        """Identify the status quo option from the research"""
        prompt = clean_indents(
            f"""
            Based on the following research for a multiple choice question, determine which option represents the status quo (the outcome if nothing changes).
            
            Research:
            {research}
            
            Question:
            {question.question_text}
            
            Options:
            {question.options}
            
            Respond with only the exact text of the option that represents the status quo. If no clear status quo exists, respond with "No clear status quo".
            """
        )
        
        try:
            response = await self.get_llm("default", "llm").invoke(prompt)
            response = response.strip()
            
            # Check if the response matches any of the options
            for option in question.options:
                if option in response:
                    return option
            
            return None
        except Exception as e:
            logger.warning(f"Error identifying status quo option: {e}")
            return None
    
    def _optimize_multiple_choice_forecasts(
        self, predictions: List[PredictedOptionList], status_quo_option: Optional[str]
    ) -> PredictedOptionList:
        """Apply domain-specific optimizations to multiple choice forecasts"""
        # Create a dictionary mapping options to lists of probabilities
        option_probabilities = {}
        for prediction in predictions:
            for option, prob in prediction.items():
                if option not in option_probabilities:
                    option_probabilities[option] = []
                option_probabilities[option].append(prob)
        
        # For each option, apply statistical optimization
        optimized_prediction = {}
        for option, probs in option_probabilities.items():
            # If this is the status quo option, give it more weight
            if status_quo_option and option == status_quo_option:
                # Calculate a weighted average that favors higher values for status quo
                sorted_probs = sorted(probs)
                # Use 60th percentile instead of median (50th) to favor status quo
                idx = int(len(sorted_probs) * 0.6)
                optimized_prediction[option] = sorted_probs[min(idx, len(sorted_probs)-1)]
            else:
                # For non-status quo options, just use median
                optimized_prediction[option] = np.median(probs)
        
        # Ensure low-probability options have at least 1% chance
        for option in optimized_prediction:
            if optimized_prediction[option] < 0.01:
                optimized_prediction[option] = 0.01
        
        # Normalize to ensure probabilities sum to 1
        total = sum(optimized_prediction.values())
        for option in optimized_prediction:
            optimized_prediction[option] /= total
        
        return optimized_prediction

    def _create_combined_multiple_choice_reasoning(
        self, 
        reasonings: List[str], 
        predictions: List[PredictedOptionList], 
        final_prediction: PredictedOptionList,
        status_quo_option: Optional[str]
    ) -> str:
        """Create combined reasoning for multiple choice questions"""
        combined = "## Statistical Analysis of Multiple Forecasts\n\n"
        combined += f"I've generated {len(predictions)} different forecasts to optimize prediction accuracy.\n\n"
        
        if status_quo_option:
            combined += f"**Status Quo Analysis**: The status quo option appears to be '{status_quo_option}'.\n\n"
        
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
            combined += f"- Final value: {final_prediction[option]:.3f}\n"
            if status_quo_option and option == status_quo_option:
                combined += f"- Note: This is the status quo option and received additional weighting.\n"
            combined += "\n"
        
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
        Generate multiple forecasts for numeric questions with uncertainty bounds optimization.
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
            
            # For some forecasts, emphasize wider uncertainty bounds
            if i < self._forecasts_per_question // 2:
                uncertainty_emphasis = "You remind yourself that good forecasters are EXTREMELY humble and set VERY wide 90/10 confidence intervals to account for unknown unknowns and surprise events."
            else:
                uncertainty_emphasis = "You remind yourself that good forecasters are humble and set appropriate 90/10 confidence intervals to account for unknown unknowns."
            
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

                {uncertainty_emphasis}

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
        
        # Apply domain-specific optimizations to widen uncertainty bounds
        optimized_prediction = self._optimize_numeric_forecasts_with_uncertainty(all_predictions, question)
        
        # Create combined reasoning
        combined_reasoning = self._create_combined_numeric_reasoning(
            all_reasonings, all_predictions, optimized_prediction
        )
        
        logger.info(
            f"Aggregated {len(all_predictions)} numeric forecasts for {question.page_url} with uncertainty optimization"
        )
        
        return ReasonedPrediction(
            prediction_value=optimized_prediction, reasoning=combined_reasoning
        )
    
    def _optimize_numeric_forecasts_with_uncertainty(
        self, predictions: List[NumericDistribution], question: NumericQuestion
    ) -> NumericDistribution:
        """Optimize numeric forecasts with uncertainty bounds adjustment"""
        # Collect all percentiles used across predictions
        all_percentiles = set()
        for prediction in predictions:
            for percentile in prediction.declared_percentiles.keys():
                all_percentiles.add(percentile)
        
        # For each percentile, take the appropriate value
        optimized_percentiles = {}
        for percentile in sorted(all_percentiles):
            values = []
            for prediction in predictions:
                if percentile in prediction.declared_percentiles:
                    values.append(prediction.declared_percentiles[percentile])
            
            if not values:
                continue
            
            # For central percentiles (40-60), use median
            if 40 <= percentile <= 60:
                optimized_percentiles[percentile] = np.median(values)
            # For lower tail percentiles (0-30), use a lower value to widen uncertainty
            elif percentile < 40:
                # Find the appropriate quantile based on percentile
                quantile = max(0.1, 0.5 - (self._uncertainty_factor * (40 - percentile) / 100))
                optimized_percentiles[percentile] = np.quantile(values, quantile)
            # For upper tail percentiles (70-100), use a higher value to widen uncertainty
            else:
                # Find the appropriate quantile based on percentile
                quantile = min(0.9, 0.5 + (self._uncertainty_factor * (percentile - 60) / 100))
                optimized_percentiles[percentile] = np.quantile(values, quantile)
        
        # Ensure monotonicity (percentiles must increase as they go up)
        sorted_percentiles = sorted(optimized_percentiles.keys())
        for i in range(1, len(sorted_percentiles)):
            curr_percentile = sorted_percentiles[i]
            prev_percentile = sorted_percentiles[i-1]
            if optimized_percentiles[curr_percentile] < optimized_percentiles[prev_percentile]:
                optimized_percentiles[curr_percentile] = optimized_percentiles[prev_percentile]
        
        # Create a new NumericDistribution with the optimized percentiles
        template_prediction = predictions[0]  # Use the first prediction as a template
        template_prediction.declared_percentiles = optimized_percentiles
        
        return template_prediction