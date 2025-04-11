import ast
import os
import typing
import yaml
from logging import getLogger

from openfl.interface.cli_helper import WORKSPACE


logger = getLogger(__name__)


class FeatureCompatibility:
    """
    Define and validate compatibility constraints for features in a plan configuration.
    This class allows specifying compatibility rules for a feature, including allowed and
    forbidden values for other features when the current feature is set to a specific value.
    It provides methods to validate a given plan configuration against these rules and
    generates warnings for any compatibility issues.
    """

    def __init__(
        self,
        feature_key: str | tuple,
        feature_value: typing.Any,
        feature_default: typing.Any,
    ):
        # Stores the name of the key that is used for the feature that has
        # compatibility constraints with other keys.
        self._feature_key = feature_key
        # Stores the value of the feature for which the compatibilit checks
        # need to happen.
        self._feature_value = feature_value
        # Stores the default value of the feature.
        # It is used if no value is set for the feature in the plan.
        # Covers parameters which are not specified in the plan yaml but set
        # in the respective class constructor.
        self._feature_default = feature_default
        # Stores a dict of keys which NEED to have a certain value when
        # feature_key is set to feature_value.
        self._allowed = {}
        # Stores a dict of keys which CANNOT to have a certain value when
        # feature_key is set to feature_value.
        self._forbidden = {}
        # Stores a list of all warning that are generated for the feature
        # compatibility.
        self._warnings = []

    def set_allowed(self, key: str | tuple, value: typing.Any):
        """
        Set a key-value pair in the internal allowed dictionary.

        Args:
            key (str | tuple): The key to set in the `_allowed` dictionary.
            value (typing.Any): The value to associate with the given key.
        """
        self._allowed[key] = value

    def set_forbidden(self, key, value):
        """
        Sets a forbidden key-value pair in the internal forbidden dictionary.

        Args:
            key (str): The key to be added to the forbidden dictionary.
            value (Any): The value associated with the key to be added.
        """
        self._forbidden[key] = value

    def validate(self, plan_config: dict) -> str:
        """
        Validates the given plan configuration against the feature's expected
        values, allowed values, and forbidden values.
        - Retrieves the feature value from the plan configuration using the
            feature key. If not present, uses the default feature value.
        - Skips validation if the feature value in the plan does not match
            the feature value required for validation.
        - Checks if all allowed values in the plan configuration are set as
            expected. Adds warnings for any discrepancies.
        - Checks for forbidden values in the plan configuration. Adds warnings
            if any forbidden values are found.
        - Returns a beautified warning message summarizing all validation
            issues, if any.

        Args:
            plan_config (dict): The configuration dictionary of the plan to validate.

        Returns:
            str: A formatted warning message if validation issues are found,
                otherwise None.
        """
        # Get feature value in plan.
        plan_value = FeatureCompatibility.get_key_value(
            self._feature_key,
            plan_config
        )
        if not plan_value:
            plan_value = self._feature_default
        logger.debug(f"Plan value of {self._feature_key} is {plan_value}")

        # Skip if the value in plan does not require validation.
        if plan_value != self._feature_value:
            return

        # Check if the all the allowed values are set as expected
        for key, value in self._allowed.items():
            # Get the value of the key in the plan.
            plan_val = FeatureCompatibility.get_key_value(
                key,
                plan_config
            )
            # Check if the value is same as allowed.
            if (isinstance(value, list) and plan_val not in value) or (plan_val != value):
                self._add_warning(key, plan_val, value, plan_value)


        # Check for the forbidden values.
        for key, value in self._forbidden.items():
            # Get the value of the key in the plan.
            plan_val = FeatureCompatibility.get_key_value(
                key,
                plan_config
            )
            # Check if the value is not one that is forbidden.
            if (isinstance(value, list) and plan_val in value) or (plan_val == value):
                self._add_warning(key, plan_val, value, plan_value, allowed_list=False)

        return self._beautify_warning()

    def _add_warning(self, plan_key, plan_value, values_list, set_value, allowed_list=True):
        """
        Adds a warning to the internal warnings list.

        Args:
            plan_key (str): The key associated with the plan being validated.
            plan_value (Any): The value associated with the plan key.
            values_list (list): A list of expected or allowed values.
            set_value (Any): The value that caused the warning.
            allowed_list (bool, optional): Indicates whether the `values_list`
                represents allowed values (True) or disallowed values (False).
                Defaults to True.
        """
        self._warnings.append(
            (plan_key, plan_value, values_list, set_value, allowed_list)
        )

    def _beautify_warning(self):
        """
        Constructs and returns a formatted warning message based on the warnings
        stored in the `_warnings` attribute. Each warning is formatted with
        specific details about the feature compatibility issue.

        Returns:
            str: A formatted string containing all the warnings, with details
                about the feature compatibility issues and suggestions for resolving them.
        """
        warn = ""
        for warning in self._warnings:
            warn += "\033[91mCannot set {} as {} when {} is set to {}.\033[0m ".format(
                FeatureCompatibility.flatten_key(warning[0]),
                warning[1],
                FeatureCompatibility.flatten_key(self._feature_key),
                warning[3]
            )
            warn += "To set {} = {}, {} must{}be set to {}.\n".format(
                FeatureCompatibility.flatten_key(self._feature_key),
                warning[3],
                FeatureCompatibility.flatten_key(warning[0]),
                " " if warning[4] else " not ",
                FeatureCompatibility.flatten_list(warning[2])
            )

        return warn

    @staticmethod
    def flatten_key(key) -> str:
        """
        Flattens a given key into a string.

        If the key is already a string, it is returned as-is. If the key is a sequence
        (e.g., a tuple or list), the elements are joined into a single string using
        a period ('.') as the separator.

        Args:
            key (str or sequence): The key to flatten. Can be a string or a sequence
                of strings.

        Returns:
            str: The flattened key as a string.
        """
        return key if isinstance(key, str) else ".".join(key)

    @staticmethod
    def flatten_list(values) -> str:
        """
        Flattens a list of strings into a single string joined by " OR ".

        If the input is a list, it concatenates the elements of the list into a
        single string, separated by " OR ". If the input is not a list, it
        returns the input as is.

        Args:
            values (list or any): A list of strings to be joined, or any other value to be
                returned as is.

        Returns:
            str: A single string if the input is a list, or the original input if it
                is not a list.
        """
        return " OR ".join(values) if isinstance(values, list) else values

    @staticmethod
    def get_key_value(key, plan_config) -> typing.Any:
        """
        Retrieve a value from a nested dictionary-like configuration based
        on the provided key.

        Args:
            key (str or tuple): The key to look up in the configuration.
                If a string, it is used directly to retrieve the value.
                If a tuple, it represents a sequence of keys to traverse
                    the nested structure.
            plan_config (dict): The dictionary-like configuration from which
                to retrieve the value.

        Returns:
            The value associated with the key in the configuration. If the key
                is a tuple and any part of the traversal fails, an empty dictionary
                is returned.
        """
        if isinstance(key, str):
            return plan_config.get(key)

        if isinstance(key, tuple):
            for index in key:
                plan_config = plan_config.get(index, {})
            return plan_config



class PlanValidation:
    """
    Validates a plan configuration against a set of predefined rules.

    Responsible for loading validation rules from a YAML configuration file, parsing them
    into a list of `FeatureCompatibility` objects, and applying these rules to a given plan
    configuration.
    """
    def __init__(self):
        # TODO: Remove this assignment.
        WORKSPACE = "/Users/pantaksh/Code/openfl_fork/openfl-workspace"
        validation_config_path = os.path.join(
            WORKSPACE, "workspace", "plan", "validations", "validation.yaml"
        )
        # Read the compatibility manifest.
        validation_config = PlanValidation.read_yaml(validation_config_path)["validation"]
        self.validation_list = []
        for key in validation_config:
            # Get the key for which compatibilti needs to be checked.
            parsed_key = PlanValidation.smart_parse(key)
            feature_dict = validation_config[key]
            # Check if a config file is provided for the key otherwise proceed.
            if feature_dict.get("config"):
                feature_config_file = os.path.join(
                    WORKSPACE, "workspace", "plan", "validations", feature_dict.get("config")
                )
                feature_dict = PlanValidation.read_yaml(feature_config_file)

            feature = FeatureCompatibility(
                parsed_key,
                feature_dict["value"],
                feature_dict["default"]
            )
            # Set the required allowed values for the feature.
            allowed_list = feature_dict.get("allowed", {})
            for allowed_key, allowed_value in allowed_list.items():
                feature.set_allowed(PlanValidation.smart_parse(allowed_key), allowed_value)
            # Set the required forbidden values for the feature.
            forbidden_list = feature_dict.get("forbidden")
            for forbidden_key, forbidden_value in forbidden_list.items():
                feature.set_forbidden(
                    PlanValidation.smart_parse(forbidden_key),
                    forbidden_value
                )

            self.validation_list.append(feature)

    def validate(self, plan_config: dict):
        """
        Validates the plan configuration using a list of validation rules.

        Iterates through the `validation_list`, applying each validation rule
        to the `_plan_config` attribute of the instance. Collects any warnings
        or messages generated by the validation rules and prints them.
        """
        warnings = ""
        for validation in self.validation_list:
            warnings += validation.validate(plan_config)

        print(warnings)

    @staticmethod
    def smart_parse(literal) -> typing.Union[tuple, str]:
        """
        Parses the input string `literal` and attempts to interpret it as a Python literal,
        a tuple-like structure, or a plain string.
        - Case 1: Tries to evaluate `literal` as a Python literal using `ast.literal_eval`.
            If successful and the result is a tuple or string, it is returned.
        - Case 2: If `literal` looks like a tuple without quotes, it is transformed into a properly quoted tuple
            and evaluated as a Python literal.
        - Case 3: If neither of the above cases apply, the input string `literal` is returned as-is.

        Args:
            literal (str): The input string to parse.

        Returns:
            Union[tuple, str]:
                - If `literal` is a valid Python literal (e.g., a tuple or string), it is returned as-is.
                - If `literal` resembles a tuple without quotes (e.g., "(a, b, c)"), it is converted into a tuple of strings.
                - Otherwise, the input string `literal` is returned unchanged.

        Example:
            >>> smart_parse("(a, b, c)")
            ('a', 'b', 'c')
            >>> smart_parse("('a', 'b', 'c')")
            ('a', 'b', 'c')
            >>> smart_parse("hello")
            'hello'
        """
        literal = literal.strip()

        # Case 1: Try to evaluate if it's a valid Python literal (like a tuple of strings)
        try:
            val = ast.literal_eval(literal)
            if isinstance(val, tuple):
                return val
            if isinstance(val, str):
                return val
        except Exception: # nosec B110
            # Skip bandit Issue: [B110:try_except_pass]
            # This is done as an exception indicates that the literal does not belong to
            # Case 1 and the flow should proceed to the next case.
            pass

        # Case 2: If it looks like a tuple without quotes: (a, b, c)
        if literal.startswith('(') and literal.endswith(')'):
            items = literal[1:-1].split(',')
            quoted_items = [f"'{item.strip()}'" for item in items if item.strip()]
            fixed = '(' + ', '.join(quoted_items) + ')'
            return ast.literal_eval(fixed)

        # Case 3: Plain string fallback
        return literal

    @staticmethod
    def read_yaml(file_path):
        yaml_contents = {}
        # Check if the file path is valid.
        if not os.path.isfile(file_path):
            raise Exception("file does not exist")

        try:
            yaml_contents = yaml.load(
                open(file_path),
                Loader=yaml.SafeLoader,
            )
        except Exception as error:
            raise error

        return yaml_contents



# TODO: Remove this block.
if __name__ == "__main__":
    # Dummy plan config.
    plan_config = {
        "aggregator": {
            "settings": {
                "secure_aggregation": True
            }
        },
        "key1": "forbidden1",
        "key2": {
            "key3": "forbidden2",
        },
    }

    # Initialise this class in `Plan` constructor.
    p = PlanValidation()
    # `validate` can be called with `initialize` method.
    p.validate(plan_config)
