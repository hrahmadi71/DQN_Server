from application import api
from flask import jsonify
from flask_restplus import Resource, fields
from application.dqn import DQN
from werkzeug.exceptions import BadRequest


# Swagger models:

get_q_values_request_body_model = api.model('DQNetInputModel', {
    'action_type': fields.Integer,
    'common_state': fields.List(fields.Float),
    'action_params': fields.List(fields.Float)
})

train_model_request_body_model = api.model('TrainModel', {
    'state': fields.List(fields.Float),
    'q_values': fields.List(fields.Float)
})

get_experience_model_request_body_model = api.model('GetExperienceModel', {
    'action_type': fields.Integer,
    'action': fields.Integer,
    'old_common_state': fields.List(fields.Float),
    'new_common_state': fields.List(fields.Float),
    'action_params': fields.List(fields.Float),
    'reward': fields.Float,
})

trained_network_request_body_model = api.model('TrainedNetworkWeightsModel', {
    'model_name': fields.String
})

possible_actions_model = api.model('PossibleActionModel', {
    'action_type': fields.Integer,
    'action_params': fields.List(fields.Integer)
})

possible_actions_list_request_body_model = api.model('PossobleActionsListModel', {
    'possible_actions': fields.List(fields.Nested(possible_actions_model))
})


# error handling functions:
def get_and_evaluate_numeric_list(data, list_name, lengths):
    desired_list = data[list_name]
    if len(desired_list) not in lengths:
        raise BadRequest("Size of '{0}' tensor suppose to be {1} but length of the q_values you sent is {2}!"
                         .format(list_name, lengths, len(desired_list)))

    if not all(isinstance(e, float) or isinstance(e, int) for e in desired_list):
        raise BadRequest("Every element of '{}' must be integer or float.".format(list_name))

    return desired_list


def get_and_evaluate(data, element_name, element_type):
    desired_element = data[element_name]

    if not isinstance(desired_element, element_type):
        raise BadRequest("Type of the '{0}' must be {1} but it is {2}!"
                         .format(element_name, element_type.__name__, type(desired_element).__name__))

    return desired_element


def get_and_evaluate_reward(data):
    desired_element = data['reward']

    if not (isinstance(desired_element, int) or isinstance(desired_element, float)):
        raise BadRequest("Reward must be numeric but it is {}!".format(type(desired_element).__name__))

    return desired_element


# APIs:
@api.route('/possible_actions/')
class PostPossibleActions(Resource):
    @api.doc(body=possible_actions_list_request_body_model)
    def post(self):
        possible_actions = []
        for action in api.payload['possible_actions']:
            possible_actions.append((action['action_type'], action['action_params']))
        DQN.set_possible_actions(possible_actions)


@api.route('/get_q_values/')
class GetQValues(Resource):
    @api.doc(body=get_q_values_request_body_model)
    def post(self):
        data = api.payload
        action_type = get_and_evaluate(data, element_name='action_type', element_type=int)
        state = get_and_evaluate_numeric_list(data, list_name='common_state', lengths=[DQN.get_common_state_regular_len()])
        action_parameters = get_and_evaluate_numeric_list(data, list_name='action_params', lengths=[6, 8, 10])
        dic = {
            'q_values': DQN.get_q_values(action_type=action_type,
                                         common_state=state,
                                         action_parameters=action_parameters)
        }
        return jsonify(dic)


@api.route('/get_experience/')
class GetExperience(Resource):
    @api.doc(body=get_experience_model_request_body_model)
    def post(self):
        data = api.payload
        action_type = get_and_evaluate(data, element_name='action_type', element_type=int)
        action = get_and_evaluate(data, element_name='action', element_type=int)
        old_common_state = get_and_evaluate_numeric_list(data,
                                                         list_name='old_common_state',
                                                         lengths=[DQN.get_common_state_regular_len()])
        new_common_state = get_and_evaluate_numeric_list(data,
                                                         list_name='new_common_state',
                                                         lengths=[DQN.get_common_state_regular_len()])
        action_parameters = get_and_evaluate_numeric_list(data,
                                                          list_name='action_params',
                                                          lengths=[6, 8, 10])
        reward = get_and_evaluate_reward(data)
        DQN.get_experience(action_type=action_type,
                           action=action,
                           old_common_state=old_common_state,
                           new_common_state=new_common_state,
                           action_parameters=action_parameters,
                           reward=reward)
        return jsonify({
            "message": "Now I've got more gray beard!"
        })


@api.route('/save_weights/')
class SaveModel(Resource):
    @api.doc(body=trained_network_request_body_model)
    def post(self):
        data = api.payload
        model_name = get_and_evaluate(data, element_name='model_name', element_type=str)
        DQN.save_model(model_name)
        return jsonify({
            'message': 'Weights saved successfully'
        })


@api.route('/load_weights/')
class SaveModel(Resource):
    @api.doc(body=trained_network_request_body_model)
    def post(self):
        data = api.payload
        model_name = get_and_evaluate(data, element_name='model_name', element_type=str)
        try:
            DQN.load_model(model_name)
            return jsonify({
                'message': 'Weights loaded successfully'
            })
        except OSError:
            return jsonify({
                'message': 'Model {} does not exist!'.format(model_name)
            })
