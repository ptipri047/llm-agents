from inspect import signature


def call_it(params: list):
    """
    Wrapping a celery task, this will attempt to create an audit log
    tying the task execution back to an organization using passed arguments

    : org_param:  str  : function parameter that denotes the organization ID
    : task_param: str  : function parameter that denotes the task instance
    """

    def decorator(func):
        """
        This is the actual task decorator, but nested to allow for parameters
        to be passed on the decorator definition
        """

        def wrapper(*args, **kwargs):
            """
            The wrapper replaces the actual function call and performs the
            needed extra auditing work before calling the original function
            """

            # create a function signature to introspect the call
            sig = signature(func)
            # Create the argument binding so we can determine what
            # parameters are given what values
            argument_binding = sig.bind(*args, **kwargs)
            argument_map = argument_binding.arguments
            
            for i,avariable in enumerate(params):
                arg = argument_map[i]
                variable = eval(avariable)
   

            # The actual logic is abstracted out to be more readable
            # Assume this function creates the relationship between the
            # task result instance and the organization instance
            assign_ownership_to_task(task, organization)

            func(*args, **kwargs)  # this calls the original function
            # as it was original intended

        return wrapper