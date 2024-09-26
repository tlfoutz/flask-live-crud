from importlib import invalidate_caches
from flask import Flask, request, jsonify, make_response, send_file
from flask_swagger_ui import get_swaggerui_blueprint
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy.orm import relationship
from os import environ
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import scipy.interpolate as scipyintpol
from scipy.integrate import quad

# Setup
#region
app = Flask(__name__)
CORS(app)
app.config['CORS_ORIGINS'] = ['file:///C:/Users/Silvertea/OneDrive/Desktop/test.html']
app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DB_URL')
db = SQLAlchemy(app)

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config = {'app_name': "Personal Garment Pattern Resource"}
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
#endregion

# DB Models
#region
class Continent(db.Model):
    __tablename__ = 'continent'

    code = db.Column(db.String(2), primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    
    child = relationship('Country', back_populates='parent')

    def json(self):
        return {
            'code': self.code,
            'name': self.name
        }
    
class Country(db.Model):
    __tablename__ = 'country'

    code = db.Column(db.String(2), primary_key=True)
    continent_code = db.Column(db.String(2), db.ForeignKey('continent.code'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    iso3 = db.Column(db.String(3), nullable=True)
    number = db.Column(db.String(3), nullable=True)
    full_name = db.Column(db.String(255), nullable=False)

    parent = relationship('Continent', back_populates='child')
    child = relationship('Person', back_populates='parent')

    def json(self):
        return {
            'code': self.code,
            'name': self.name,
            'continent_code': self.continent_code,
            'iso3': self.iso3,
            'number': self.number,
            'full_name': self.full_name
        }
    
class Person(db.Model):
    __tablename__ = 'person'

    id = db.Column(db.Integer, primary_key=True)
    birth_year = db.Column(db.Integer, nullable=False)
    is_male = db.Column(db.Boolean, nullable=False)
    is_metric = db.Column(db.Boolean, nullable=False)
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    email = db.Column(db.String(64), unique=True, nullable=False)
    country_code = db.Column(db.String(2), db.ForeignKey('country.code'), nullable=False)
    
    parent = relationship('Country', back_populates='child')
    child1 = relationship('BodyMeasurements', back_populates='parent', cascade='all, delete-orphan')
    child2 = relationship('PatternPoints', back_populates='parent', cascade='all, delete-orphan')

    def json(self):
        return {
            'id': self.id,
            'birth_year': self.birth_year,
            'is_male': self.is_male,
            'is_metric': self.is_metric,
            'height': self.height,
            'weight': self.weight,
            'email': self.email,
            'country_code': self.country_code
        }
    
class BodyMeasurements(db.Model):
    __tablename__ = 'body_measurements'

    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('person.id'), unique=True, nullable=False)
    neck = db.Column(db.Float)
    bust = db.Column(db.Float)
    chest = db.Column(db.Float)
    waist = db.Column(db.Float)
    abdomen = db.Column(db.Float)
    hip = db.Column(db.Float)
    center_length_front = db.Column(db.Float)
    center_length_back = db.Column(db.Float)
    full_length_front = db.Column(db.Float)
    full_length_back = db.Column(db.Float)
    shoulder_slope_front = db.Column(db.Float)
    shoulder_slope_back = db.Column(db.Float)
    new_strap = db.Column(db.Float)
    bust_depth = db.Column(db.Float)
    bust_radius = db.Column(db.Float)
    bust_span = db.Column(db.Float)
    side_length = db.Column(db.Float)
    neck_front = db.Column(db.Float)
    neck_back = db.Column(db.Float)
    shoulder_length = db.Column(db.Float)
    across_shoulder_front = db.Column(db.Float)
    across_shoulder_back = db.Column(db.Float)
    across_front = db.Column(db.Float)
    across_back = db.Column(db.Float)
    bust_arc = db.Column(db.Float)
    back_arc = db.Column(db.Float)
    waist_arc_front = db.Column(db.Float)
    waist_arc_back = db.Column(db.Float)
    abdomen_arc_front = db.Column(db.Float)
    abdomen_arc_back = db.Column(db.Float)
    hip_arc_front = db.Column(db.Float)
    hip_arc_back = db.Column(db.Float)
    hip_depth_front = db.Column(db.Float)
    hip_depth_side = db.Column(db.Float)
    hip_depth_back = db.Column(db.Float)
    knee_length = db.Column(db.Float)
    ankle_length = db.Column(db.Float)
    inseam = db.Column(db.Float)
    floor_length = db.Column(db.Float)
    crotch_length = db.Column(db.Float)
    crotch_depth = db.Column(db.Float)
    arm_length = db.Column(db.Float)
    elbow_length = db.Column(db.Float)
    cap_height = db.Column(db.Float)
    bicep = db.Column(db.Float)
    wrist = db.Column(db.Float)
    hand = db.Column(db.Float)
    thigh = db.Column(db.Float)
    knee = db.Column(db.Float)
    calf = db.Column(db.Float)
    ankle = db.Column(db.Float)
    
    parent = relationship('Person', back_populates='child1')

    def json(self):
        return {
            'id': self.id,
            'person_id': self.person_id,
            'neck': self.neck,
            'bust': self.bust,
            'chest': self.chest,
            'waist': self.waist,
            'abdomen': self.abdomen,
            'center_length_front': self.center_length_front,
            'center_length_back': self.center_length_back,
            'full_length_front': self.full_length_front,
            'full_length_back': self.full_length_back,
            'shoulder_slope_front': self.shoulder_slope_front,
            'shoulder_slope_back': self.shoulder_slope_back,
            'new_strap': self.new_strap,
            'bust_depth': self.bust_depth,
            'bust_radius': self.bust_radius,
            'bust_span': self.bust_span,
            'side_length': self.side_length,
            'neck_front': self.neck_front,
            'neck_back': self.neck_back,
            'shoulder_length': self.shoulder_length,
            'across_shoulder_front': self.across_shoulder_front,
            'across_shoulder_back': self.across_shoulder_back,
            'across_front': self.across_front,
            'across_back': self.across_back,
            'bust_arc': self.bust_arc,
            'back_arc': self.back_arc,
            'waist_arc_front': self.waist_arc_front,
            'waist_arc_back': self.waist_arc_back,
            'abdomen_arc_front': self.abdomen_arc_front,
            'abdomen_arc_back': self.abdomen_arc_back,
            'hip_arc_front': self.hip_arc_front,
            'hip_arc_back': self.hip_arc_back,
            'hip_depth_front': self.hip_depth_front,
            'hip_depth_side': self.hip_depth_side,
            'hip_depth_back': self.hip_depth_back,
            'knee_length': self.knee_length,
            'ankle_length': self.ankle_length,
            'inseam': self.inseam,
            'floor_length': self.floor_length,
            'crotch_length': self.crotch_length,
            'crotch_depth': self.crotch_depth,
            'arm_length': self.arm_length,
            'elbow_length': self.elbow_length,
            'cap_height': self.cap_height,
            'bicep': self.bicep,
            'wrist': self.wrist,
            'hand': self.hand,
            'thigh': self.thigh,    
            'knee': self.knee,
            'calf': self.calf,
            'ankle': self.ankle
        }
    
class PatternPoints(db.Model):
    __tablename__ = 'pattern_points'

    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('person.id'), unique=True, nullable=False)
    center_front_neck_x = db.Column(db.Float)
    center_front_neck_y = db.Column(db.Float)
    center_back_neck_x = db.Column(db.Float)
    center_back_neck_y = db.Column(db.Float)
    center_front_waist_x = db.Column(db.Float)
    center_front_waist_y = db.Column(db.Float)
    center_back_waist_x = db.Column(db.Float)
    center_back_waist_y = db.Column(db.Float)
    high_shoulder_x = db.Column(db.Float)
    high_shoulder_y = db.Column(db.Float)
    shoulder_tip_x = db.Column(db.Float)
    shoulder_tip_y = db.Column(db.Float)
    side_waist_x = db.Column(db.Float)
    side_waist_y = db.Column(db.Float)
    bust_point_x = db.Column(db.Float)
    bust_point_y = db.Column(db.Float)
    sternum_x = db.Column(db.Float)
    sternum_y = db.Column(db.Float)
    side_abdomen_x = db.Column(db.Float)
    side_abdomen_y = db.Column(db.Float)
    side_hip_x = db.Column(db.Float)
    side_hip_y = db.Column(db.Float)
    below_armpit_x = db.Column(db.Float)
    below_armpit_y = db.Column(db.Float)
    front_axillary_fold_x = db.Column(db.Float)
    front_axillary_fold_y = db.Column(db.Float)
    back_axillary_fold_x = db.Column(db.Float)
    back_axillary_fold_y = db.Column(db.Float)
    
    parent = relationship('Person', back_populates='child2')

    def json(self):
        return {
            'id': self.id,
            'person_id': self.person_id,
            'center_front_neck_x': self.center_front_neck_x,
            'center_front_neck_y': self.center_front_neck_y,
            'center_back_neck_x': self.center_back_neck_x,
            'center_back_neck_y': self.center_back_neck_y,
            'center_front_waist_x': self.center_front_waist_x,
            'center_front_waist_y': self.center_front_waist_y,
            'center_back_waist_x': self.center_back_waist_x,
            'center_back_waist_y': self.center_back_waist_y,
            'high_shoulder_x': self.high_shoulder_x,
            'high_shoulder_y': self.high_shoulder_y,
            'shoulder_tip_x': self.shoulder_tip_x,
            'shoulder_tip_y': self.shoulder_tip_y,
            'side_waist_x': self.side_waist_x,
            'side_waist_y': self.side_waist_y,
            'bust_point_x': self.bust_point_x,
            'bust_point_y': self.bust_point_y,
            'sternum_x': self.sternum_x,
            'sternum_y': self.sternum_y,
            'side_abdomen_x': self.side_abdomen_x,
            'side_abdomen_y': self.side_abdomen_y,
            'side_hip_x': self.side_hip_x,
            'side_hip_y': self.side_hip_y,
            'below_armpit_x': self.below_armpit_x,
            'below_armpit_y': self.below_armpit_y,
            'front_axillary_fold_x': self.front_axillary_fold_x,
            'front_axillary_fold_y': self.front_axillary_fold_y,
            'back_axillary_fold_x': self.back_axillary_fold_x,
            'back_axillary_fold_y': self.back_axillary_fold_y
        }
    
class Sizes(db.Model):
    __tablename__ = 'sizes'

    id = db.Column(db.Integer, primary_key=True)
    origin_name = db.Column(db.String(32), nullable=False)
    description = db.Column(db.String(128), nullable=False)
    size_symbol = db.Column(db.String(8))
    body_measurement = db.Column(db.String(32))
    min_value = db.Column(db.Float)
    max_value = db.Column(db.Float)
    source_url = db.Column(db.String(256))

    def json(self):
        return {
            'id': self.id,
            'origin_name': self.origin_name,
            'description': self.description,
            'size_symbol': self.size_symbol,
            'body_measurement': self.body_measurement,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'source_url': self.source_url
        }
    
db.create_all()
#endregion

# Person POST / GET / PUT
#region
# create person
@app.route('/person', methods=['POST'])
def create_person():
  try:
    data = request.get_json()
    new_user = Person(birth_year=data['birth_year'], is_male=data['is_male'], is_metric=data['is_metric'], 
                      height=data['height'], weight=data['weight'], email=data['email'], country_code=data['country_code'])
    db.session.add(new_user)
    db.session.commit()
    return make_response(jsonify({'message': 'user created'}), 201)
  except Exception as e:
    return make_response(jsonify({'message': 'error creating user'}), 500)

# get person by id
@app.route('/person/<int:id>', methods=['GET'])
def get_person(id):
  try:
    person = Person.query.filter_by(id=id).first()
    if person:
      return make_response(jsonify({'person': person.json()}), 200)
    return make_response(jsonify({'message': 'person not found'}), 404)
  except Exception as e:
    return make_response(jsonify({'message': 'error getting person'}), 500)

# update person
@app.route('/person/<int:id>', methods=['PUT'])
def update_person(id):
  try:
    person = Person.query.filter_by(id=id).first()
    if person:
      data = request.get_json()
      person.birth_year = data['birth_year']
      person.is_male = data['is_male']
      person.is_metric = data['is_metric']
      person.height = data['height']
      person.weight = data['weight']
      person.email = data['email']
      person.country_code = data['country_code']
      db.session.commit()
      return make_response(jsonify({'message': 'user updated'}), 200)
    return make_response(jsonify({'message': 'user not found'}), 404)
  except Exception as e:
    return make_response(jsonify({'message': 'error updating user'}), 500)

#endregion

# BodyMeasurements POST / GET / PUT
#region
# create body measurements
@app.route('/body_measurements', methods=['POST'])
def create_body_measurements():
  try:
    data = request.get_json()
    new_bm = BodyMeasurements(
       person_id=data['person_id'],
       neck=data['neck'],
       bust=data['bust'],
       chest=data['chest'],
       waist=data['waist'],
       abdomen=data['abdomen'],
       hip=data['hip'],
       center_length_front=data['center_length_front'],
       center_length_back=data['center_length_back'],
       full_length_front=data['full_length_front'],
       full_length_back=data['full_length_back'],
       shoulder_slope_front=data['shoulder_slope_front'],
       shoulder_slope_back=data['shoulder_slope_back'],
       new_strap=data['new_strap'],
       bust_depth=data['bust_depth'],
       bust_radius=data['bust_radius'],
       bust_span=data['bust_span'],
       side_length=data['side_length'],
       neck_front=data['neck_front'],
       neck_back=data['neck_back'],
       shoulder_length=data['shoulder_length'],
       across_shoulder_front=data['across_shoulder_front'],
       across_shoulder_back=data['across_shoulder_back'],
       across_front=data['across_front'],
       across_back=data['across_back'],
       bust_arc=data['bust_arc'],
       back_arc=data['back_arc'],
       waist_arc_front=data['waist_arc_front'],
       waist_arc_back=data['waist_arc_back'],
       abdomen_arc_front=data['abdomen_arc_front'],
       abdomen_arc_back=data['abdomen_arc_back'],
       hip_arc_front=data['hip_arc_front'],
       hip_arc_back=data['hip_arc_back'],
       hip_depth_front=data['hip_depth_front'],
       hip_depth_side=data['hip_depth_side'],
       hip_depth_back=data['hip_depth_back'],
       knee_length=data['knee_length'],
       ankle_length=data['ankle_length'],
       inseam=data['inseam'],
       floor_length=data['floor_length'],
       crotch_length=data['crotch_length'],
       crotch_depth=data['crotch_depth'],
       arm_length=data['arm_length'],
       elbow_length=data['elbow_length'],
       cap_height=data['cap_height'],
       bicep=data['bicep'],
       wrist=data['wrist'],
       hand=data['hand'],
       thigh=data['thigh'],
       knee=data['knee'],
       calf=data['calf'],
       ankle=data['ankle'])
    db.session.add(new_bm)
    db.session.commit()
    return make_response(jsonify({'message': 'body measurements added'}), 201)
  except Exception as e:
    return make_response(jsonify({'message': 'error adding body measurements'}), 500)

# get body measurements by id
@app.route('/body_measurements/<int:id>', methods=['GET'])
def get_body_measurements(id):
  try:
    bm = BodyMeasurements.query.filter_by(person_id=id).first()
    if bm:
      return make_response(jsonify({'body_measurements': bm.json()}), 200)
    return make_response(jsonify({'message': 'body measurements not found'}), 404)
  except Exception as e:
    return make_response(jsonify({'message': 'error getting body measurements'}), 500)

# update body measurements
@app.route('/body_measurements/<int:id>', methods=['PUT'])
def update_body_measurements(id):
  try:
    bm = BodyMeasurements.query.filter_by(person_id=id).first()
    if bm:
      data = request.get_json()
      bm.neck = data['neck']
      bm.bust = data['bust']
      bm.chest = data['chest']
      bm.waist = data['waist']
      bm.abdomen = data['abdomen']
      bm.hip = data['hip']
      bm.center_length_front = data['center_length_front']
      bm.center_length_back = data['center_length_back']
      bm.full_length_front = data['full_length_front']
      bm.full_length_back = data['full_length_back']
      bm.shoulder_slope_front = data['shoulder_slope_front']
      bm.shoulder_slope_back = data['shoulder_slope_back']
      bm.new_strap = data['new_strap']
      bm.bust_depth = data['bust_depth']
      bm.bust_radius = data['bust_radius']
      bm.bust_span = data['bust_span']
      bm.side_length = data['side_length']
      bm.neck_front = data['neck_front']
      bm.neck_back = data['neck_back']
      bm.shoulder_length = data['shoulder_length']
      bm.across_shoulder_front = data['across_shoulder_front']
      bm.across_shoulder_back = data['across_shoulder_back']
      bm.across_front = data['across_front']
      bm.across_back = data['across_back']
      bm.bust_arc = data['bust_arc']
      bm.back_arc = data['back_arc']
      bm.waist_arc_front = data['waist_arc_front']
      bm.waist_arc_back = data['waist_arc_back']
      bm.abdomen_arc_front = data['abdomen_arc_front']
      bm.abdomen_arc_back = data['abdomen_arc_back']
      bm.hip_arc_front = data['hip_arc_front']
      bm.hip_arc_back = data['hip_arc_back']
      bm.hip_depth_front = data['hip_depth_front']
      bm.hip_depth_side = data['hip_depth_side']
      bm.hip_depth_back = data['hip_depth_back']
      bm.knee_length = data['knee_length']
      bm.ankle_length = data['ankle_length']
      bm.inseam = data['inseam']
      bm.floor_length = data['floor_length']
      bm.crotch_length = data['crotch_length']
      bm.crotch_depth = data['crotch_depth']
      bm.arm_length = data['arm_length']
      bm.elbow_length = data['elbow_length']
      bm.cap_height = data['cap_height']
      bm.bicep = data['bicep']
      bm.wrist = data['wrist']
      bm.hand = data['hand']
      bm.thigh = data['thigh']
      bm.knee = data['knee']
      bm.calf = data['calf']
      bm.ankle = data['ankle']
      db.session.commit()
      return make_response(jsonify({'message': 'body measurements updated'}), 200)
    return make_response(jsonify({'message': 'body measurements not found'}), 404)
  except Exception as e:
    return make_response(jsonify({'message': 'error updating body measurements'}), 500)

#endregion  

# PatternPoints POST / GET / PUT
#region
# create pattern points
@app.route('/pattern_points', methods=['POST'])
def create_pattern_points():
  try:
    data = request.get_json()
    new_pp = PatternPoints(
       person_id=data['person_id'],
       center_front_neck_x=data['center_front_neck_x'],
       center_front_neck_y=data['center_front_neck_y'],
       center_back_neck_x=data['center_back_neck_x'],
       center_back_neck_y=data['center_back_neck_y'],
       center_front_waist_x=data['center_front_waist_x'],
       center_front_waist_y=data['center_front_waist_y'],
       center_back_waist_x=data['center_back_waist_x'],
       center_back_waist_y=data['center_back_waist_y'],
       high_shoulder_x=data['high_shoulder_x'],
       high_shoulder_y=data['high_shoulder_y'],
       shoulder_tip_x=data['shoulder_tip_x'],
       shoulder_tip_y=data['shoulder_tip_y'],
       side_waist_x=data['side_waist_x'],
       side_waist_y=data['side_waist_y'],
       bust_point_x=data['bust_point_x'],
       bust_point_y=data['bust_point_y'],
       sternum_x=data['sternum_x'],
       sternum_y=data['sternum_y'],
       side_abdomen_x=data['side_abdomen_x'],
       side_abdomen_y=data['side_abdomen_y'],
       side_hip_x=data['side_hip_x'],
       side_hip_y=data['side_hip_y'],
       below_armpit_x=data['below_armpit_x'],
       below_armpit_y=data['below_armpit_y'],
       front_axillary_fold_x=data['front_axillary_fold_x'],
       front_axillary_fold_y=data['front_axillary_fold_y'],
       back_axillary_fold_x=data['back_axillary_fold_x'],
       back_axillary_fold_y=data['back_axillary_fold_y'])
    db.session.add(new_pp)
    db.session.commit()
    return make_response(jsonify({'message': 'pattern points added'}), 201)
  except Exception as e:
    return make_response(jsonify({'message': 'error adding pattern points'}), 500)

# get pattern points by id
@app.route('/pattern_points/<int:id>', methods=['GET'])
def get_pattern_points(id):
  try:
    pp = PatternPoints.query.filter_by(person_id=id).first()
    if pp:
      return make_response(jsonify({'pattern_points': pp.json()}), 200)
    return make_response(jsonify({'message': 'pattern points not found'}), 404)
  except Exception as e:
    return make_response(jsonify({'message': 'error getting pattern points'}), 500)

# update pattern points
@app.route('/pattern_points/<int:id>', methods=['PUT'])
def update_pattern_points(id):
  try:
    pp = PatternPoints.query.filter_by(person_id=id).first()
    if pp:
      data = request.get_json()
      pp.center_front_neck_x = data['center_front_neck_x']
      pp.center_front_neck_y = data['center_front_neck_y']
      pp.center_back_neck_x = data['center_back_neck_x']
      pp.center_back_neck_y = data['center_back_neck_y']
      pp.center_front_waist_x = data['center_front_waist_x']
      pp.center_front_waist_y = data['center_front_waist_y']
      pp.center_back_waist_x = data['center_back_waist_x']
      pp.center_back_waist_y = data['center_back_waist_y']
      pp.high_shoulder_x = data['high_shoulder_x']
      pp.high_shoulder_y = data['high_shoulder_y']
      pp.shoulder_tip_x = data['shoulder_tip_x']
      pp.shoulder_tip_y = data['shoulder_tip_y']
      pp.side_waist_x = data['side_waist_x']
      pp.side_waist_y = data['side_waist_y']
      pp.bust_point_x = data['bust_point_x']
      pp.bust_point_y = data['bust_point_y']
      pp.sternum_x = data['sternum_x']
      pp.sternum_y = data['sternum_y']
      pp.side_abdomen_x = data['side_abdomen_x']
      pp.side_abdomen_y = data['side_abdomen_y']
      pp.side_hip_x = data['side_hip_x']
      pp.side_hip_y = data['side_hip_y']
      pp.below_armpit_x = data['below_armpit_x']
      pp.below_armpit_y = data['below_armpit_y']
      pp.front_axillary_fold_x = data['front_axillary_fold_x']
      pp.front_axillary_fold_y = data['front_axillary_fold_y']
      pp.back_axillary_fold_x = data['back_axillary_fold_x']
      pp.back_axillary_fold_y = data['back_axillary_fold_y']
      db.session.commit()
      return make_response(jsonify({'message': 'pattern points updated'}), 200)
    return make_response(jsonify({'message': 'pattern points not found'}), 404)
  except Exception as e:
    return make_response(jsonify({'message': 'error updating pattern points'}), 500)

#endregion

# Pattern Drafting GET
#region
  
@app.route('/pattern/mens/sloper/torso/front/<int:id>', methods=['GET'])
def get_mens_sloper_torso_front(id):
    bm = BodyMeasurements.query.filter_by(person_id=id).first()
    return mensSloperTorsoFront(bm)

@app.route('/pattern/mens/sloper/torso/back/<int:id>', methods=['GET'])
def get_mens_sloper_torso_back(id):
    bm = BodyMeasurements.query.filter_by(person_id=id).first()
    return mensSloperTorsoBack(bm)

@app.route('/pattern/womens/sloper/bodice/front/<int:id>', methods=['GET'])
def get_womens_sloper_bodice_front(id):
    bm = BodyMeasurements.query.filter_by(person_id=id).first()
    return womensSloperBodiceFront(bm)

@app.route('/pattern/womens/sloper/bodice/back/<int:id>', methods=['GET'])
def get_womens_sloper_bodice_back(id):
    bm = BodyMeasurements.query.filter_by(person_id=id).first()
    return womensSloperBodiceBack(bm)

@app.route('/pattern/womens/sloper/skirt/front/<int:id>', methods=['GET'])
def get_womens_sloper_skirt_front(id):
    bm = BodyMeasurements.query.filter_by(person_id=id).first()
    return womensSloperSkirtFront(bm)

@app.route('/pattern/womens/sloper/skirt/back/<int:id>', methods=['GET'])
def get_womens_sloper_skirt_back(id):
    bm = BodyMeasurements.query.filter_by(person_id=id).first()
    return womensSloperSkirtBack(bm)

@app.route('/pattern/womens/sloper/sleeve/<int:id>', methods=['GET'])
def get_womens_sloper_sleeve(id):
    bm = BodyMeasurements.query.filter_by(person_id=id).first()
    return womensSloperSleeve(bm)

@app.route('/pattern/mens/sloper/sleeve/<int:id>', methods=['GET'])
def get_mens_sloper_sleeve(id):
    bm = BodyMeasurements.query.filter_by(person_id=id).first()
    return mensSloperSleeve(bm)

@app.route('/pattern/unisex/sloper/pants/front/<int:id>', methods=['GET'])
def get_unisex_sloper_pants_front(id):
    bm = BodyMeasurements.query.filter_by(person_id=id).first()
    return unisexSloperPantFront(bm)

#endregion

# mensSloper
#region

def mensSloperTorsoFront(bm):
    points = mensTorsoPoints(bm, front=True, sleeve=False)
    
    Path = mpath.Path
    fig, ax = plt.subplots()
    pp1 = mpatches.PathPatch(
        Path([points["E"], points["F"], points["J"], points["J2"], points["J3"], points["L2"], points["G3"],
              points["G"], points["H"], points["A3"], points["A2"], points["A"], points["E"]],
             [Path.MOVETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.LINETO,
              Path.LINETO,
              Path.CURVE3,
              Path.CURVE3,
              Path.LINETO,
              Path.LINETO]),
    fc="none", transform=ax.transData, linewidth=1)
    
    ax.add_patch(pp1)
    
    # Set Axis
    height = points["M"][1]
    width = max(points["D"][0], points["G"][0], points["F"][0], points["J"][0])
    setAxis(ax, fig, height, width)
    
    # Save PDF
    name = mensSloperTorsoFront.__name__
    return savePDF(plt, name)

def mensSloperTorsoBack(bm):
    points = mensTorsoPoints(bm, front=False, sleeve=False)

    Path = mpath.Path
    fig, ax = plt.subplots()
    pp1 = mpatches.PathPatch(
        Path([points["E"], points["F"], points["J"], points["J2"], points["J3"], points["L2"], points["G3"],
              points["G"], points["H"], points["C3"], points["C2"], points["C"], points["E"]],
             [Path.MOVETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.LINETO,
              Path.LINETO,
              Path.CURVE3,
              Path.CURVE3,
              Path.LINETO,
              Path.LINETO]),
    fc="none", transform=ax.transData, linewidth=1)
    
    ax.add_patch(pp1)
        
    # Set Axis
    height = points["M"][1]
    width = max(points["D"][0], points["G"][0], points["F"][0], points["J"][0])
    setAxis(ax, fig, height, width)
    
    # Save PDF
    name = mensSloperTorsoBack.__name__
    return savePDF(plt, name)

def mensSloperSleeve(bm):
    N = (0,0)
    P = (N[0], N[0] + bm.arm_length)
    M = (P[0], P[1] - bm.bicep/3)
    O = (N[0], P[1] - bm.elbow_length)
    backArmhole = mensTorsoPoints(bm, front=False, sleeve=True) - 0.5
    RX = math.sqrt(pow(backArmhole, 2) - pow(P[1] - M[1], 2))
    R = (M[0] - RX, M[1])
    one = ((R[0] - M[0]) * 0.875, M[1] + (P[1] - M[1]) * 0.125)
    four = ((R[0] - M[0]) * 0.5, M[1] + (P[1] - M[1]) * 0.5)
    six = ((R[0] - M[0]) * 0.25, M[1] + (P[1] - M[1]) * 0.75)
    capeSlope = (P[1] - R[1])/(P[0] - R[0])
    invCapeSlope = -(1/capeSlope)
    xChange = 0.25/math.sqrt(1 + pow(invCapeSlope, 2))
    yChange = invCapeSlope * xChange
    one2 = (one[0] + xChange, one[1] + yChange)
    xChange = 0.375/math.sqrt(1 + pow(invCapeSlope, 2))
    yChange = invCapeSlope * xChange
    four2 = (four[0] - xChange, four[1] - yChange)
    xChange = 0.75/math.sqrt(1 + pow(invCapeSlope, 2))
    yChange = invCapeSlope * xChange
    six2 = (six[0] - xChange, six[1] - yChange)
    frontArmhole = mensTorsoPoints(bm, front=True, sleeve=True) - 0.25
    SX = math.sqrt(pow(frontArmhole, 2) - pow(P[1] - M[1], 2))
    S = (M[0] + SX, M[1])
    nine = ((S[0] - M[0]) * 0.25, M[1] + (P[1] - M[1]) * 0.75)
    ten = ((S[0] - M[0]) * 0.5, M[1] + (P[1] - M[1]) * 0.5)
    eleven = ((S[0] - M[0]) * 0.75, M[1] + (P[1] - M[1]) * 0.25)
    capeSlope = (S[1] - P[1])/(S[0] - P[0])
    invCapeSlope = -(1/capeSlope)
    xChange = 0.625/math.sqrt(1 + pow(invCapeSlope, 2))
    yChange = invCapeSlope * xChange
    nine2 = (nine[0] + xChange, nine[1] + yChange)
    xChange = 0.125/math.sqrt(1 + pow(invCapeSlope, 2))
    yChange = invCapeSlope * xChange
    ten2 = (ten[0] + xChange, ten[1] + yChange)
    xChange = 0.5/math.sqrt(1 + pow(invCapeSlope, 2))
    yChange = invCapeSlope * xChange
    eleven2 = (eleven[0] - xChange, eleven[1] - yChange)
    six2P = (six2[0], P[1])
    midsix2four2 = ((six2[0] + four2[0])/2 + (four2[0] - six2[0])/4, (six2[1] + four2[1])/2)
    four2one2 = ((four2[0] + one2[0])/2, one2[1])
    midone2R = ((one2[0] + R[0])/2, (one2[1] + R[1])/2)
    nine2P = (nine2[0], P[1])
    midnine2ten2 = ((nine2[0] + ten2[0])/2 + (ten2[0] - nine2[0])/4, (nine2[1] + ten2[1])/2)
    ten2eleven2 = ((ten2[0] + eleven2[0])/2, eleven2[1])
    mideleven2S = ((eleven2[0] + S[0])/2, (eleven2[1] + S[1])/2)
    T = (bm.wrist/2 + 1, N[1]) # 1 inch loose around wrist
    U = (-(bm.wrist/2 + 1), T[1]) # 1 inch loose around wrist

    Path = mpath.Path
    fig, ax = plt.subplots()
    pp1 = mpatches.PathPatch(
        Path([T, S, mideleven2S, ten2eleven2, ten2, midnine2ten2, nine2P, P, six2P, midsix2four2, four2, four2one2, midone2R, R, U, T],
             [Path.MOVETO,
              Path.LINETO,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,  
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,  
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.LINETO,
              Path.LINETO]),
    fc="none", transform=ax.transData, linewidth=1)
   
    ax.add_patch(pp1)
     
    # points = {"N":N, "P":P, "M":M, "O":O, "R":R, "one":one, "four":four, "six":six, "one2":one2, "four2":four2, "six2":six2,
    #           "S":S, "nine":nine, "ten":ten, "eleven":eleven, "nine2":nine2, "ten2":ten2, "eleven2":eleven2,
    #           "six2P":six2P, "midsix2four2":midsix2four2, "four2one2":four2one2, "midone2R":midone2R,
    #           "nine2P":nine2P, "midnine2ten2":midnine2ten2, "ten2eleven2": ten2eleven2, "mideleven2S":mideleven2S,
    #           "T":T, "U":U}
    
    # testingPoints(points)
   
    # Set Axis
    height = P[1]
    width = S[0] - R[0]
    setAxis(ax, fig, height, width)

    # Save PDF
    name = mensSloperSleeve.__name__
    return savePDF(plt, name)

def mensTorsoPoints(bm, front, sleeve):
    E = (0,0) # center hip
    B = (E[0], E[1] + bm.hip_depth_side) # center waist
    A = (E[0], B[1] + bm.center_length_front) # center neck front
    C = (E[0], B[1] + bm.center_length_back) # center neck back
    M = (E[0], B[1] + bm.full_length_front) # center top high shoulder
    I = (E[0], B[1] + bm.side_length) # center below armpit
    K = (E[0], (C[1] + I[1])/2) # center neck back / below armpit midpoint
    D = (E[0] + bm.waist/4 + 1.25, B[1]) # side waist
    F = (E[0] + bm.hip/4 + 1.25, E[1]) # side hip
    J = (E[0] + bm.chest/4 + 1.25, I[1]) # side below armpit
    J2 = (J[0] - 0.25, J[1]) # to square off for armscye
    if front:
       N = (E[0] + bm.across_shoulder_front, M[1]) # top low shoulder
       G = (N[0], B[1] + math.sqrt(pow(bm.shoulder_slope_front, 2) - pow(N[0], 2))) # low shoulder
       H = (N[0] - math.sqrt(pow(bm.shoulder_length, 2) - pow((N[1] - G[1]),2)), N[1]) # high shoulder
       lowShoulderSquaredOffPoints = lowShoulderSquaredOff(N, G, H)
       G2 = lowShoulderSquaredOffPoints["ControlPoint"]
       G3 = lowShoulderSquaredOffPoints["SquaredOffPoint"]
       L = (E[0] + bm.across_front, K[1]) # armscye
       L2 = (L[0] - 1, L[1]) # armscye bezier control - BIG TODO: calcuate so bezier curve line actually on L
       J3 = (L[0], J2[1]) # armscye bezier control
       if sleeve:
        return (cubicBezierLen(J2, J3, L2, G3) + 0.5) # add K2-J and G3-G lengths
       A2 = (A[0] + 0.25, A[1]) # to square off front neckline curve
       A3 = (H[0], A[1]) # front neck control point
       return {"A":A, "A2":A2, "A3":A3, "B":B, "C":C, "D":D, "E":E, "F":F, "G":G, "G2":G2, "G3":G3, "H":H, "I":I,
               "J":J, "J2":J2, "J3":J3, "L":L, "L2":L2, "M":M, "N":N}
    else:
       N = (E[0] + bm.across_shoulder_back, M[1]) # top low shoulder
       G = (N[0], B[1] + math.sqrt(pow(bm.shoulder_slope_back, 2) - pow(N[0], 2))) # low shoulder
       H = (N[0] - math.sqrt(pow(bm.shoulder_length, 2) - pow((N[1] - G[1]),2)), N[1]) # high shoulder
       lowShoulderSquaredOffPoints = lowShoulderSquaredOff(N, G, H)
       G2 = lowShoulderSquaredOffPoints["ControlPoint"]
       G3 = lowShoulderSquaredOffPoints["SquaredOffPoint"]
       L = (E[0] + bm.across_back, K[1]) # armscye
       L2 = (L[0] - 1, L[1]) # armscye bezier control - BIG TODO: calcuate so bezier curve line actually on L
       J3 = (L[0], J2[1]) # armscye bezier control
       if sleeve:
        return (cubicBezierLen(J2, J3, L2, G3) + 0.5) # add K2-J and G3-G lengths
       C2 = (C[0] + 0.25, C[1]) # to square off back neckline curve
       C3 = (H[0], C[1]) # back neck control point
       return {"A":A, "B":B, "C":C, "C2":C2, "C3":C3, "D":D, "E":E, "F":F, "G":G, "G2":G2, "G3":G3, "H":H, "I":I,
               "J":J, "J2":J2, "J3":J3, "L":L, "L2":L2, "M":M, "N":N}
   
#endregion

# womensSloper  
#region
    
def womensSloperBodiceFront(bm):
    # B = (0,0) # center waist
    # Y = (B[0], B[1] + bm.full_length_front + 0.125) # center top high shoulder, plus 1/8"
    # X = (Y[0] + bm.across_shoulder_front + 0.125, Y[1]) # top low shoulder, plus 1/8"
    # A = (B[0], B[1] + bm.center_length_front) # center neck front
    # A2 = (A[0] + 0.25, A[1])
    # Z = (B[0] + bm.bust_arc + 0.25, B[1]) # bottom below armpit, total of 1/2" bust level ease
    # D = (X[0], math.sqrt(pow(bm.shoulder_slope_front + 0.125, 2) - pow(X[0], 2))) # low shoulder
    # directVectBD = (D[0] - B[0], D[1] - B[1])
    # BD = math.sqrt(pow(directVectBD[0], 2) + pow(directVectBD[1], 2))
    # normDirectVectBD = (directVectBD[0]/BD, directVectBD[1]/BD)
    # D2 = (D[0] - (0.25 * normDirectVectBD[0]), D[1] - (0.25 * normDirectVectBD[1]))
    # BU = bm.shoulder_slope_front + 0.125 - bm.bust_depth # length B to U
    # U = (D[0] - (BU * D[0]/BD), D[1] - (BU * D[1]/BD)) # bust depth
    # V = (B[0], U[1]) # center bust depth
    # C = (math.sqrt(pow(bm.shoulder_length, 2) - pow(X[1] - D[1], 2)), X[1]) # high shoulder
    # slopeCD = (D[1] - C[1])/(D[0] - C[0])
    # invSlopeCD = -(1/slopeCD) # perpendicular to slopeCD, so negative reciprocal
    # C3 = ((A[1] - C[1])/invSlopeCD + C[0], A[1])
    # C2 = (C[0], A[1] + (C[1] - A[1]) * (C3[0]/C[0])) # makeshift bezier control point
    # H = (B[0] + bm.bust_span + 0.25, V[1])
    # W = (A[0], (A[1] + V[1])/2)
    # T = (B[0] + bm.across_front + 0.25, W[1])
    # T2 = (T[0] - 0.5, T[1]) # armscye bezier control - BIG TODO: calcuate so bezier curve line actually on T
    # R = (B[0] + bm.bust_span, B[1])
    # R2 = (R[0], R[1] - 0.1875)
    # S = (Z[0], C[1] - math.sqrt(pow(bm.new_strap + 0.125, 2) - pow(Z[0] - C[0], 2)))
    # K = (S[0], B[1] + bm.side_length)
    # E = (S[0] + 1.25, S[1])
    # directVectEK = (K[0] - E[0], K[1] - E[1])
    # EK = math.sqrt(pow(directVectEK[0], 2) + pow(directVectEK[1], 2))
    # normDirectVectEK = (directVectEK[0]/EK, directVectEK[1]/EK)
    # posPerpDirectVectEK = (normDirectVectEK[1], -normDirectVectEK[0])
    # K2 = (K[0] + -0.25 * posPerpDirectVectEK[0], K[1] + -0.25 + posPerpDirectVectEK[1])
    # slopeKK2 = (K[1] - K2[1])/(K[0] - K2[0])
    # T3 = (T[0], slopeKK2 * (T[0] - K[0]) + K[1])
    # EP = bm.waist_arc_front + 0.25 - R[0]
    # directVectER2 = (E[0] - R2[0], E[1] - R2[1])
    # ER2 = math.sqrt(pow(directVectER2[0], 2) + pow(directVectER2[1], 2))
    # P = (E[0] - EP * directVectER2[0]/ER2, E[1] - EP * directVectER2[1]/ER2)
    # HR2 = math.sqrt(pow(R2[0] - H[0], 2) + pow(R2[1] - H[1], 2))
    # directVectHP = (P[0] - H[0], P[1] - H[1])
    # HP = math.sqrt(pow(directVectHP[0], 2) + pow(directVectHP[1], 2))
    # normDirectVectHP = (directVectHP[0]/HP, directVectHP[1]/HP)
    # Q = (H[0] + HR2 * normDirectVectHP[0], H[1] + HR2 * normDirectVectHP[1])
    # midQR2 = ((Q[0] + R2[0])/2, (Q[1] + R2[1])/2)
    # vectHMid = (midQR2[0] - H[0], midQR2[1] - H[1])
    # HMid = math.sqrt(pow(vectHMid[0], 2) + pow(vectHMid[1], 2))
    # normVectHMid = (vectHMid[0]/HMid, vectHMid[1]/HMid)
    # H2 = (H[0] + (0.625 * normVectHMid[0]), H[1] + (0.625 * normVectHMid[1]))
    # midEQ = ((E[0] + Q[0])/2,  (E[1] + Q[1])/2)
    # directVectEQ = (E[0] - Q[0], E[1] - Q[1])
    # EQ = math.sqrt(pow(directVectEQ[0], 2) + pow(directVectEQ[1], 2))
    # normDirectVectEQ = (directVectEQ[0]/EQ, directVectEQ[1]/EQ)
    # posPerpDirectVectEQ = (normDirectVectEQ[1], -normDirectVectEQ[1])
    # midEQ2 = (midEQ[0] + (10 * (posPerpDirectVectEQ[0])), midEQ[1] + (10 * (posPerpDirectVectEQ[1])))
    # midBR2 = ((B[0] + R2[0])/2, (B[1] + R2[1])/2)
    # directVectBR2 = (B[0] - R2[0], B[1] - R2[1])
    # BR2 = math.sqrt(pow(directVectBR2[0], 2) + pow(directVectBR2[1], 2))
    # normDirectVectBR2 = (directVectBR2[0]/BR2, directVectBR2[1]/BR2)
    # posPerpDirectVectBR2 = (normDirectVectBR2[1], -normDirectVectBR2[1])
    # midBR22 = (midBR2[0] + (-3 * (posPerpDirectVectBR2[0])), midBR2[1] + (-3 * (posPerpDirectVectBR2[1])))
    points = womensTorsoPoints(bm, front=True, sleeve=False)

    Path = mpath.Path
    fig, ax = plt.subplots()
    pp1 = mpatches.PathPatch(
        Path([points["B"], points["midBR22"], points["R2"], points["H2"], points["Q"], points["midEQ2"],
              points["E"], points["K"], points["K2"], points["T3"], points["T2"], points["D2"], points["D"],
              points["C"], points["C2"], points["C3"], points["A2"], points["A"], points["B"]],
             [Path.MOVETO,
              Path.CURVE3,
              Path.CURVE3,
              Path.LINETO,
              Path.LINETO,
              Path.CURVE3,
              Path.CURVE3,
              Path.LINETO,
              Path.LINETO,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.LINETO,
              Path.LINETO,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.LINETO,
              Path.LINETO]),
    fc="none", transform=ax.transData, linewidth=1)
    
    ax.add_patch(pp1)
    
    # Set Axis
    height = points["Y"][1] - points["R2"][1]
    width = points["E"][0]
    setAxis(ax, fig, height, width)

    # Save PDF
    name = womensSloperBodiceFront.__name__
    return savePDF(plt, name)

def womensSloperBodiceBack(bm):
   # G = (0,0)
   # Z = (G[0], bm.full_length_back)
   # Y = (G[0] + bm.across_shoulder_back, Z[1])
   # F = (G[0], G[1] + bm.center_length_back)
   # F2 = (F[0] + 0.25, F[1])
   # N = (G[0] + bm.back_arc + 0.75, G[1])
   # C = (Z[0] + bm.neck_back + 0.125, Z[1])
   # V = (Y[0], math.sqrt(pow(bm.shoulder_slope_back + 0.125, 2) - pow(Y[0], 2)))
   # directVectCV = (V[0] - C[0], V[1] - C[1])
   # CV = math.sqrt(pow(directVectCV[0], 2) + pow(directVectCV[1], 2))
   # normDirectVectCV = (directVectCV[0]/CV, directVectCV[1]/CV)
   # D = (C[0] + (bm.shoulder_length + 0.5) * normDirectVectCV[0], C[1] + (bm.shoulder_length + 0.5) * normDirectVectCV[1])
   # slopeCD = (D[1] - C[1])/(D[0] - C[0])
   # invSlopeCD = -(1/slopeCD) # perpendicular to slopeCD, so negative reciprocal
   # xChangeD = 0.25/math.sqrt(1 + pow(invSlopeCD, 2))
   # yChangeD = invSlopeCD * xChangeD
   # D2 = (D[0] - xChangeD, D[1] - yChangeD)
   # C3 = ((F[1] - C[1])/invSlopeCD + C[0], F[1])
   # C2 = (C[0], F[1] + (C[1] - F[1]) * (C3[0]/C[0])) # makeshift bezier control point
   # Q = (G[0] + bm.bust_span, G[1])
   # P = (G[0] + bm.waist_arc_back + 1.75, G[0])  # 1.75" = dart intake of 1.5" and 1/4" ease
   # S = (Q[0] + 1.5, Q[1]) # 1.5" dart intake
   # R = ((Q[0] + S[0])/2, (Q[1] + S[1])/2) # QS midpoint
   # E = (P[0], P[1] - 0.1875)
   # K = (N[0], E[1] + math.sqrt(pow(bm.side_length, 2) - pow(E[0] - N[0], 2)))
   # slopeEK = (E[1] - K[1])/(E[0] - K[0])
   # invSlopeEK = -(1/slopeEK) # perpendicular to slopeEK, so negative reciprocal
   # xChangeK = 0.25/math.sqrt(1 + pow(invSlopeEK, 2))
   # yChangeK = invSlopeEK * xChangeK
   # K2 = (K[0] - xChangeK, K[1] - yChangeK)
   # slopeKK2 = (K[1] - K2[1])/(K[0] - K2[0])
   # T = (R[0], R[1] + bm.side_length - 1)
   # directVectQT = (Q[0] - T[0], Q[1] - T[1])
   # QT = math.sqrt(pow(directVectQT[0], 2) + pow(directVectQT[1], 2))
   # normDirectVectQT = (directVectQT[0]/QT, directVectQT[1]/QT)
   # Q2 = (T[0] + (QT + 0.125) * normDirectVectQT[0], T[1] + (QT + 0.125) * normDirectVectQT[1])
   # S2 = (R[0] + R[0] - Q2[0], Q2[1]) # symmetrical dart legs
   # X = ((C[0] + D[0])/2, (C[1] + D[1])/2) # CD midpoint
   # directVectXT = (T[0] - X[0], T[1] - X[1])
   # XT = math.sqrt(pow(directVectXT[0], 2) + pow(directVectXT[1], 2))
   # normDirectVectXT = (directVectXT[0]/XT, directVectQT[1]/XT)
   # U = (X[0] + 3 * normDirectVectXT[0], X[1] + 3 * normDirectVectXT[1])
   # directVectCD = (D[0] - C[0], D[1] - C[1])
   # CD = math.sqrt(pow(directVectCD[0], 2) + pow(directVectCD[1], 2))
   # normDirectVectCD = (directVectCD[0]/CD, directVectCD[1]/CD)
   # X2 = (C[0] + (CD/2 - 0.25) * normDirectVectCD[0], C[1] + (CD/2 - 0.25) * normDirectVectCD[1])
   # X3 = (C[0] + (CD/2 + 0.25) * normDirectVectCD[0], C[1] + (CD/2 + 0.25) * normDirectVectCD[1])
   # xChangeX2 = 0.125/math.sqrt(1 + pow(invSlopeCD, 2))
   # yChangeX2 = invSlopeCD * xChangeX2
   # X2_2 = (X2[0] + xChangeX2, X2[1] + yChangeX2) # dart leg high shoulder side
   # X3_2 = (X3[0] + X2_2[0] - X2[0], X3[1] + X2_2[1] - X2[1]) # dart leg low shoulder side
   # O = (F[0], F[1] - F[1] * 0.25)
   # A = (O[0] + bm.across_back + 0.25, O[1])
   # A2 = (A[0] - 0.5, A[1]) # armscye bezier control - BIG TODO: calcuate so bezier curve line actually on A
   # K3 = (A[0], K[1])
   # midGQ2 = ((G[0] + Q2[0])/2,  (G[1] + Q2[1])/2)
   # directVectGQ2 = (G[0] - Q2[0], G[1] - Q2[1])
   # GQ2 = math.sqrt(pow(directVectGQ2[0], 2) + pow(directVectGQ2[1], 2))
   # normDirectVectGQ2 = (directVectGQ2[0]/GQ2, directVectGQ2[1]/GQ2)
   # posPerpDirectVectGQ2 = (normDirectVectGQ2[1], -normDirectVectGQ2[1])
   # midGQ22 = (midGQ2[0] + (10 * (posPerpDirectVectGQ2[0])), midGQ2[1] + (-10 * (posPerpDirectVectGQ2[1])))
   # midS2E = ((S2[0] + E[0])/2,  (S2[1] + E[1])/2)
   # directVectS2E = (S2[0] - E[0], S2[1] - E[1])
   # S2E = math.sqrt(pow(directVectS2E[0], 2) + pow(directVectS2E[1], 2))
   # normDirectVectS2E = (directVectS2E[0]/S2E, directVectS2E[1]/S2E)
   # posPerpDirectVectS2E = (normDirectVectS2E[1], -normDirectVectS2E[1])
   # midS2E2 = (midS2E[0] + (10 * (posPerpDirectVectS2E[0])), midS2E[1] + (-10 * (posPerpDirectVectS2E[1])))
   points = womensTorsoPoints(bm, front=False, sleeve=False)

   Path = mpath.Path
   fig, ax = plt.subplots()
   pp1 = mpatches.PathPatch(
        Path([points["G"], points["midGQ22"], points["Q2"], points["T"], points["S2"], points["midS2E2"],
              points["E"], points["K"], points["K2"], points["K3"], points["A2"], points["D2"], points["D"],
              points["X3_2"], points["U"], points["X2_2"], points["C"], points["C2"], points["C3"],
              points["F2"], points["F"], points["G"]],
             [Path.MOVETO,
              Path.CURVE3,
              Path.CURVE3,
              Path.LINETO,
              Path.LINETO,
              Path.CURVE3,
              Path.CURVE3,
              Path.LINETO,
              Path.LINETO,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.LINETO,
              Path.LINETO]),
   fc="none", transform=ax.transData, linewidth=1)
   
   ax.add_patch(pp1)
   
   # Set Axis
   height = points["Z"][1] - points["Q"][1]
   width = max(points["E"][0], points["N"][0])
   setAxis(ax, fig, height, width)

   # Save PDF
   name = womensSloperBodiceBack.__name__
   return savePDF(plt, name)

def womensSloperSleeve(bm):
    frontBodiceArmhole = womensTorsoPoints(bm, front=True, sleeve=True)
    backBodiceArmhole = womensTorsoPoints(bm, front=False, sleeve=True)
    armhole = (frontBodiceArmhole + backBodiceArmhole)/2 + 0.25
    Y = (0,0)
    D = (Y[0], Y[1] + bm.arm_length)
    W = (D[0], D[1] - bm.cap_height)
    M1 = (W[0], (Y[1] + W[1])/2)
    M2 = (M1[0], M1[1] + 0.75)
    one = (-math.sqrt(pow(armhole, 2) - pow(D[1] - W[1], 2)), W[1])
    two = (-bm.bicep/2, W[1])
    KB = (two[0] - 1, two[1])
    KF = (-KB[0], KB[1])
    Q = ((KB[0] - W[0]) * 0.75, W[1] + (D[1] - W[1]) * 0.25)
    R = ((KB[0] - W[0]) * 0.5, W[1] + (D[1] - W[1]) * 0.5)
    S = ((KB[0] - W[0]) * 0.25, W[1] + (D[1] - W[1]) * 0.75)
    T = (-S[0], S[1])
    U = (-R[0], R[1])
    V = (-Q[0], Q[1])
    X = (KB[0] + 2, Y[1])
    Z = (-X[0], X[1])
    capeSlope = (D[1] - KB[1])/(D[0] - KB[0])
    invCapeSlope = -(1/capeSlope)
    xChange = 0.375/math.sqrt(1 + pow(invCapeSlope, 2))
    yChange = invCapeSlope * xChange
    Q2 = (Q[0] + xChange, Q[1] + yChange)
    xChange = 0.25/math.sqrt(1 + pow(invCapeSlope, 2))
    yChange = invCapeSlope * xChange
    R2 = (R[0] - xChange, R[1] - yChange)
    xChange = 0.625/math.sqrt(1 + pow(invCapeSlope, 2))
    yChange = invCapeSlope * xChange
    S2 = (S[0] - xChange, S[1] - yChange)
    xChange = 0.75/math.sqrt(1 + pow(-invCapeSlope, 2))
    yChange = -invCapeSlope * xChange
    T2 = (T[0] + xChange, T[1] + yChange)    
    xChange = 0.1875/math.sqrt(1 + pow(-invCapeSlope, 2))
    yChange = -invCapeSlope * xChange
    U2 = (U[0] + xChange, U[1] + yChange)
    xChange = 0.5/math.sqrt(1 + pow(-invCapeSlope, 2))
    yChange = -invCapeSlope * xChange
    V2 = (V[0] - xChange, V[1] - yChange)
    slopeKFZ = (KF[1] - Z[1])/(KF[0] - Z[0])
    T2D = (T2[0], D[1])
    midT2U2 = ((T2[0] + U2[0])/2 + (U2[0] - T2[0])/4, (T2[1] + U2[1])/2)
    U2V2 = ((U2[0] + V2[0])/2, V2[1])
    midV2KF = ((V2[0] + KF[0])/2, (V2[1] + KF[1])/2)
    S2D = (S2[0], D[1])
    midS2R2 = ((S2[0] + R2[0])/2 + (R2[0] - S2[0])/4, (S2[1] + R2[1])/2)
    R2Q2 = ((R2[0] + Q2[0])/2, Q2[1])
    midQ2KB = ((Q2[0] + KB[0])/2, (Q2[1] + KB[1])/2)
    L = ((M2[1] - Z[1])/slopeKFZ + Z[0], M2[1])
    C = (-L[0] - 0.25, L[1])
    N = (C[0], C[1] - 1)
    O = ((C[0] - M2[0])/2, (C[1] + M2[1])/2)
    directVectON = (N[0] - O[0], N[1] - O[1])
    ON = math.sqrt(pow(directVectON[0], 2) + pow(directVectON[1], 2))
    normDirectVectON = (directVectON[0]/ON, directVectON[1]/ON)
    N2 = (O[0] - (C[0] - O[0]) * normDirectVectON[0], O[1] - (C[0] - O[0]) * normDirectVectON[1])
    P = (X[0] + 0.75, X[1])
    directVectNP = (P[0] - N[0], P[1] - N[1])
    NP = math.sqrt(pow(directVectNP[0], 2) + pow(directVectNP[1], 2))
    normDirectVectNP = (directVectNP[0]/NP, directVectNP[1]/NP)
    LZ = math.sqrt(pow(Z[0] - L[0], 2) + pow(Z[1] - L[1], 2))
    B = (N[0] + LZ * normDirectVectNP[0], N[1] + LZ * normDirectVectNP[1])
    directVectBZ = (B[0] - Z[0], B[1] - Z[1])
    BZ = math.sqrt(pow(directVectBZ[0], 2) + pow(directVectBZ[1], 2))
    normDirectVectBZ = (directVectBZ[0]/BZ, directVectBZ[1]/BZ)
    XZ = X[0] - Z[0]
    A = (B[0] + XZ * normDirectVectBZ[0], B[1] + XZ * normDirectVectBZ[1])

    Path = mpath.Path
    fig, ax = plt.subplots()
    pp1 = mpatches.PathPatch(
        Path([B, A, L, KF, midV2KF, U2V2, U2, midT2U2, T2D, D, S2D, midS2R2, R2, R2Q2, midQ2KB, KB, C, O, N2, B],
             [Path.MOVETO,
              Path.LINETO,
              Path.CURVE3,
              Path.CURVE3,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,  
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,  
              Path.CURVE4,
              Path.CURVE4,
              Path.CURVE4,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO]),
    fc="none", transform=ax.transData, linewidth=1)
   
    ax.add_patch(pp1)
     
    # points = {"Y":Y, "D":D, "W":W, "M1":M1, "M2":M2, "one":one, "two":two, "KB":KB, "KF":KF, "Q":Q, "R":R, "S":S,
    #           "T":T, "U":U, "V":V, "X":X, "Z":Z, "Q2":Q2, "R2":R2, "S2":S2, "T2":T2, "U2":U2, "V2":V2, "L":L, "C":C,
    #           "O":O, "N":N, "N2":N2, "P":P, "B":B, "A":A, "T2D":T2D, "U2V2":U2V2, "midT2U2":midT2U2, "midV2KF":midV2KF,
    #           "S2D":S2D, "R2Q2":R2Q2, "midS2R2":midS2R2, "midQ2KB":midQ2KB}
    
    # testingPoints(points)
   
    # Set Axis
    height = D[1]
    width = KF[0] - KB[0]
    setAxis(ax, fig, height, width)

    # Save PDF
    name = womensSloperSleeve.__name__
    return savePDF(plt, name)

def womensTorsoPoints(bm, front, sleeve):
    if front:
        B = (0,0) # center waist
        Y = (B[0], B[1] + bm.full_length_front + 0.125) # center top high shoulder, plus 1/8"
        X = (Y[0] + bm.across_shoulder_front + 0.125, Y[1]) # top low shoulder, plus 1/8"
        A = (B[0], B[1] + bm.center_length_front) # center neck front
        A2 = (A[0] + 0.25, A[1])
        Z = (B[0] + bm.bust_arc + 0.25, B[1]) # bottom below armpit, total of 1/2" bust level ease
        D = (X[0], math.sqrt(pow(bm.shoulder_slope_front + 0.125, 2) - pow(X[0], 2))) # low shoulder
        directVectBD = (D[0] - B[0], D[1] - B[1])
        BD = math.sqrt(pow(directVectBD[0], 2) + pow(directVectBD[1], 2))
        normDirectVectBD = (directVectBD[0]/BD, directVectBD[1]/BD)
        D2 = (D[0] - (0.25 * normDirectVectBD[0]), D[1] - (0.25 * normDirectVectBD[1]))
        BU = bm.shoulder_slope_front + 0.125 - bm.bust_depth # length B to U
        U = (D[0] - (BU * D[0]/BD), D[1] - (BU * D[1]/BD)) # bust depth
        V = (B[0], U[1]) # center bust depth
        C = (math.sqrt(pow(bm.shoulder_length, 2) - pow(X[1] - D[1], 2)), X[1]) # high shoulder
        slopeCD = (D[1] - C[1])/(D[0] - C[0])
        invSlopeCD = -(1/slopeCD) # perpendicular to slopeCD, so negative reciprocal
        C3 = ((A[1] - C[1])/invSlopeCD + C[0], A[1])
        C2 = (C[0], A[1] + (C[1] - A[1]) * (C3[0]/C[0])) # makeshift bezier control point
        H = (B[0] + bm.bust_span + 0.25, V[1])
        W = (A[0], (A[1] + V[1])/2)
        T = (B[0] + bm.across_front + 0.25, W[1])
        T2 = (T[0] - 0.5, T[1]) # armscye bezier control - BIG TODO: calcuate so bezier curve line actually on T
        R = (B[0] + bm.bust_span, B[1])
        R2 = (R[0], R[1] - 0.1875)
        S = (Z[0], C[1] - math.sqrt(pow(bm.new_strap + 0.125, 2) - pow(Z[0] - C[0], 2)))
        K = (S[0], B[1] + bm.side_length)
        E = (S[0] + 1.25, S[1])
        directVectEK = (K[0] - E[0], K[1] - E[1])
        EK = math.sqrt(pow(directVectEK[0], 2) + pow(directVectEK[1], 2))
        normDirectVectEK = (directVectEK[0]/EK, directVectEK[1]/EK)
        posPerpDirectVectEK = (normDirectVectEK[1], -normDirectVectEK[0])
        K2 = (K[0] + -0.25 * posPerpDirectVectEK[0], K[1] + -0.25 + posPerpDirectVectEK[1])
        slopeKK2 = (K[1] - K2[1])/(K[0] - K2[0])
        T3 = (T[0], slopeKK2 * (T[0] - K[0]) + K[1])
        if sleeve:
            return (cubicBezierLen(K2, T3, T2, D2) + 0.5) # add K2-K and D2-D lengths
        EP = bm.waist_arc_front + 0.25 - R[0]
        directVectER2 = (E[0] - R2[0], E[1] - R2[1])
        ER2 = math.sqrt(pow(directVectER2[0], 2) + pow(directVectER2[1], 2))
        P = (E[0] - EP * directVectER2[0]/ER2, E[1] - EP * directVectER2[1]/ER2)
        HR2 = math.sqrt(pow(R2[0] - H[0], 2) + pow(R2[1] - H[1], 2))
        directVectHP = (P[0] - H[0], P[1] - H[1])
        HP = math.sqrt(pow(directVectHP[0], 2) + pow(directVectHP[1], 2))
        normDirectVectHP = (directVectHP[0]/HP, directVectHP[1]/HP)
        Q = (H[0] + HR2 * normDirectVectHP[0], H[1] + HR2 * normDirectVectHP[1])
        midQR2 = ((Q[0] + R2[0])/2, (Q[1] + R2[1])/2)
        vectHMid = (midQR2[0] - H[0], midQR2[1] - H[1])
        HMid = math.sqrt(pow(vectHMid[0], 2) + pow(vectHMid[1], 2))
        normVectHMid = (vectHMid[0]/HMid, vectHMid[1]/HMid)
        H2 = (H[0] + (0.625 * normVectHMid[0]), H[1] + (0.625 * normVectHMid[1]))
        midEQ = ((E[0] + Q[0])/2,  (E[1] + Q[1])/2)
        directVectEQ = (E[0] - Q[0], E[1] - Q[1])
        EQ = math.sqrt(pow(directVectEQ[0], 2) + pow(directVectEQ[1], 2))
        normDirectVectEQ = (directVectEQ[0]/EQ, directVectEQ[1]/EQ)
        posPerpDirectVectEQ = (normDirectVectEQ[1], -normDirectVectEQ[1])
        midEQ2 = (midEQ[0] + (10 * (posPerpDirectVectEQ[0])), midEQ[1] + (10 * (posPerpDirectVectEQ[1])))
        midBR2 = ((B[0] + R2[0])/2, (B[1] + R2[1])/2)
        directVectBR2 = (B[0] - R2[0], B[1] - R2[1])
        BR2 = math.sqrt(pow(directVectBR2[0], 2) + pow(directVectBR2[1], 2))
        normDirectVectBR2 = (directVectBR2[0]/BR2, directVectBR2[1]/BR2)
        posPerpDirectVectBR2 = (normDirectVectBR2[1], -normDirectVectBR2[1])
        midBR22 = (midBR2[0] + (-3 * (posPerpDirectVectBR2[0])), midBR2[1] + (-3 * (posPerpDirectVectBR2[1])))
        return {"B":B, "midBR22":midBR22, "R2":R2, "H2":H2, "Q":Q, "midEQ2":midEQ2, "E":E, "K":K, "K2":K2, "T3":T3,
                "T2":T2, "D2":D2, "D":D, "C":C, "C2":C2, "C3":C3, "A2":A2, "A":A, "Y":Y, "R2":R2, "E":E}
    else:
        G = (0,0)
        Z = (G[0], bm.full_length_back)
        Y = (G[0] + bm.across_shoulder_back, Z[1])
        F = (G[0], G[1] + bm.center_length_back)
        F2 = (F[0] + 0.25, F[1])
        N = (G[0] + bm.back_arc + 0.75, G[1])
        C = (Z[0] + bm.neck_back + 0.125, Z[1])
        V = (Y[0], math.sqrt(pow(bm.shoulder_slope_back + 0.125, 2) - pow(Y[0], 2)))
        directVectCV = (V[0] - C[0], V[1] - C[1])
        CV = math.sqrt(pow(directVectCV[0], 2) + pow(directVectCV[1], 2))
        normDirectVectCV = (directVectCV[0]/CV, directVectCV[1]/CV)
        D = (C[0] + (bm.shoulder_length + 0.5) * normDirectVectCV[0], C[1] + (bm.shoulder_length + 0.5) * normDirectVectCV[1])
        slopeCD = (D[1] - C[1])/(D[0] - C[0])
        invSlopeCD = -(1/slopeCD) # perpendicular to slopeCD, so negative reciprocal
        xChangeD = 0.25/math.sqrt(1 + pow(invSlopeCD, 2))
        yChangeD = invSlopeCD * xChangeD
        D2 = (D[0] - xChangeD, D[1] - yChangeD)
        C3 = ((F[1] - C[1])/invSlopeCD + C[0], F[1])
        C2 = (C[0], F[1] + (C[1] - F[1]) * (C3[0]/C[0])) # makeshift bezier control point
        Q = (G[0] + bm.bust_span, G[1])
        P = (G[0] + bm.waist_arc_back + 1.75, G[0])  # 1.75" = dart intake of 1.5" and 1/4" ease
        S = (Q[0] + 1.5, Q[1]) # 1.5" dart intake
        R = ((Q[0] + S[0])/2, (Q[1] + S[1])/2) # QS midpoint
        E = (P[0], P[1] - 0.1875)
        K = (N[0], E[1] + math.sqrt(pow(bm.side_length, 2) - pow(E[0] - N[0], 2)))
        slopeEK = (E[1] - K[1])/(E[0] - K[0])
        invSlopeEK = -(1/slopeEK) # perpendicular to slopeEK, so negative reciprocal
        xChangeK = 0.25/math.sqrt(1 + pow(invSlopeEK, 2))
        yChangeK = invSlopeEK * xChangeK
        K2 = (K[0] - xChangeK, K[1] - yChangeK)
        slopeKK2 = (K[1] - K2[1])/(K[0] - K2[0])
        T = (R[0], R[1] + bm.side_length - 1)
        directVectQT = (Q[0] - T[0], Q[1] - T[1])
        QT = math.sqrt(pow(directVectQT[0], 2) + pow(directVectQT[1], 2))
        normDirectVectQT = (directVectQT[0]/QT, directVectQT[1]/QT)
        Q2 = (T[0] + (QT + 0.125) * normDirectVectQT[0], T[1] + (QT + 0.125) * normDirectVectQT[1])
        S2 = (R[0] + R[0] - Q2[0], Q2[1]) # symmetrical dart legs
        X = ((C[0] + D[0])/2, (C[1] + D[1])/2) # CD midpoint
        directVectXT = (T[0] - X[0], T[1] - X[1])
        XT = math.sqrt(pow(directVectXT[0], 2) + pow(directVectXT[1], 2))
        normDirectVectXT = (directVectXT[0]/XT, directVectQT[1]/XT)
        U = (X[0] + 3 * normDirectVectXT[0], X[1] + 3 * normDirectVectXT[1])
        directVectCD = (D[0] - C[0], D[1] - C[1])
        CD = math.sqrt(pow(directVectCD[0], 2) + pow(directVectCD[1], 2))
        normDirectVectCD = (directVectCD[0]/CD, directVectCD[1]/CD)
        X2 = (C[0] + (CD/2 - 0.25) * normDirectVectCD[0], C[1] + (CD/2 - 0.25) * normDirectVectCD[1])
        X3 = (C[0] + (CD/2 + 0.25) * normDirectVectCD[0], C[1] + (CD/2 + 0.25) * normDirectVectCD[1])
        xChangeX2 = 0.125/math.sqrt(1 + pow(invSlopeCD, 2))
        yChangeX2 = invSlopeCD * xChangeX2
        X2_2 = (X2[0] + xChangeX2, X2[1] + yChangeX2) # dart leg high shoulder side
        X3_2 = (X3[0] + X2_2[0] - X2[0], X3[1] + X2_2[1] - X2[1]) # dart leg low shoulder side
        O = (F[0], F[1] - F[1] * 0.25)
        A = (O[0] + bm.across_back + 0.25, O[1])
        A2 = (A[0] - 0.5, A[1]) # armscye bezier control - BIG TODO: calcuate so bezier curve line actually on A
        K3 = (A[0], K[1])
        if sleeve:
            return (cubicBezierLen(K2, K3, A2, D2) + 0.5) # add K2-K and D2-D lengths
        midGQ2 = ((G[0] + Q2[0])/2,  (G[1] + Q2[1])/2)
        directVectGQ2 = (G[0] - Q2[0], G[1] - Q2[1])
        GQ2 = math.sqrt(pow(directVectGQ2[0], 2) + pow(directVectGQ2[1], 2))
        normDirectVectGQ2 = (directVectGQ2[0]/GQ2, directVectGQ2[1]/GQ2)
        posPerpDirectVectGQ2 = (normDirectVectGQ2[1], -normDirectVectGQ2[1])
        midGQ22 = (midGQ2[0] + (10 * (posPerpDirectVectGQ2[0])), midGQ2[1] + (-10 * (posPerpDirectVectGQ2[1])))
        midS2E = ((S2[0] + E[0])/2,  (S2[1] + E[1])/2)
        directVectS2E = (S2[0] - E[0], S2[1] - E[1])
        S2E = math.sqrt(pow(directVectS2E[0], 2) + pow(directVectS2E[1], 2))
        normDirectVectS2E = (directVectS2E[0]/S2E, directVectS2E[1]/S2E)
        posPerpDirectVectS2E = (normDirectVectS2E[1], -normDirectVectS2E[1])
        midS2E2 = (midS2E[0] + (10 * (posPerpDirectVectS2E[0])), midS2E[1] + (-10 * (posPerpDirectVectS2E[1])))
        return {"G":G, "midGQ22":midGQ22, "Q2":Q2, "T":T, "S2":S2, "midS2E2":midS2E2, "E":E, "K":K, "K2":K2, "K3":K3,
                "A2":A2, "D2":D2, "D":D, "X3_2":X3_2, "U":U, "X2_2":X2_2, "C":C, "C2":C2, "C3":C3, "F2":F2, "F":F, "Z":Z,
                "Q":Q, "E":E, "N":N}

def womensSloperSkirtFront(bm):
   front = True
   points = womensSkirtPoints(bm, front)

   Path = mpath.Path
   fig, ax = plt.subplots()
   pp1 = mpatches.PathPatch(
        Path([points["S"], points["R"], points["B"], points["U"], points["UU2"], points["U2"], points["U3"], points["U3U4"],
              points["U4"], points["EF2"], points["EF"], points["IF3"], points["I2"], points["I"], points["S"]],
             [Path.MOVETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.CURVE3,
              Path.CURVE3,
              Path.CURVE3,
              Path.CURVE3,
              Path.LINETO,
              Path.LINETO]),
   fc="none", transform=ax.transData, linewidth=1)
   
   ax.add_patch(pp1)

   # Set Axis
   height = points["EF"][1]
   width = points["R"][0]
   setAxis(ax, fig, height, width)

   # Save PDF
   name = womensSloperSkirtFront.__name__
   return savePDF(plt, name)

def womensSloperSkirtBack(bm):
   front = False
   points = womensSkirtPoints(bm, front)

   Path = mpath.Path
   fig, ax = plt.subplots()
   pp1 = mpatches.PathPatch(
        Path([points["S"], points["Q"], points["G"], points["D"], points["DD2"], points["D2"], points["D3"], points["D3D4"],
              points["D4"], points["EB2"], points["EB"], points["IB3"], points["I2"], points["I"], points["S"]],
             [Path.MOVETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.CURVE3,
              Path.CURVE3,
              Path.CURVE3,
              Path.CURVE3,
              Path.LINETO,
              Path.LINETO]),
   fc="none", transform=ax.transData, linewidth=1)    
   
   ax.add_patch(pp1)
   
   # Set Axis
   height = points["EB"][1]
   width = -points["Q"][0]
   setAxis(ax, fig, height, width)

   # Save PDF
   name = womensSloperSkirtBack.__name__
   return savePDF(plt, name)

def womensSkirtPoints(bm, front):
   S = (0,0)
   T = (S[0], S[1] + bm.knee_length)
   I = (T[0], T[1] - bm.hip_depth_front)
   I2 = (I[0], I[1] + (T[1] - I[1])/3)
   if front:
    B = (T[0] + bm.hip_arc_front + 0.5, T[1])
    Z = (B[0], I[1])
    R = (B[0], S[0])
    dartsF = dartSkirt(bm.hip - bm.waist, True)
    V = (B[0] - (bm.waist_arc_front + (dartsF["dartIntake"] * dartsF["dartNum"])), B[1])
    U = (B[0] - bm.bust_span, B[1])
    U2 = (U[0] - dartsF["dartIntake"], U[1])
    U3 = (U2[0] - 1.25, U2[1])
    if dartsF["dartNum"] == 2:
     U4 = (U3[0] - dartsF["dartIntake"], U3[1])
    else:
     U4 = (U3[0], U3[1])
    directVectIV = (V[0] - I[0], V[1] - I[1])
    IV = math.sqrt(pow(directVectIV[0], 2) + pow(directVectIV[1], 2))
    if bm.hip_depth_side < bm.hip_depth_front:
     IEF = bm.hip_depth_side + 2 * (bm.hip_depth_front - bm.hip_depth_side)  
    else:
     IEF = bm.hip_depth_side
    EF = (I[0] + IEF * directVectIV[0]/IV, I[1] + IEF * directVectIV[1]/IV)
    UU2 = ((U[0] + U2[0])/2, U[1] - 3.5)
    U3U4 = ((U3[0] + U4[0])/2, U[1] - 3.5)
    EF2 = ((EF[0] + U4[0])/2, U4[1])
    IF3 = (I2[0], (EF[1] + I2[1])/2)
    return {"S":S, "R":R, "B":B, "U":U, "UU2":UU2, "U2":U2, "U3":U3, "U3U4":U3U4, "U4":U4, "EF2":EF2, "EF":EF, "IF3":IF3, "I2":I2, "I":I}
   else:
    X = (T[0] - bm.hip_arc_back - 0.5, T[1])
    Y = (X[0], I[1])
    Q = (X[0], S[0])
    G = (Y[0], Y[1] + bm.hip_depth_back)
    dartsB = dartSkirt(bm.hip - bm.waist, False)
    W = (X[0] + bm.waist_arc_back + (dartsB["dartIntake"] * dartsB["dartNum"]), X[1])
    directVectIW = (W[0] - I[0], W[1] - I[1])
    IW = math.sqrt(pow(directVectIW[0], 2) + pow(directVectIW[1], 2))
    if bm.hip_depth_side < bm.hip_depth_front:
     IEB = bm.hip_depth_side + 2 * (bm.hip_depth_front - bm.hip_depth_side)  
    else:
     IEB = bm.hip_depth_side
    EB = (I[0] + IEB * directVectIW[0]/IW, I[1] + IEB * directVectIW[1]/IW)
    GW = math.sqrt(pow(W[0] - G[0], 2) + pow(W[1] - G[1], 2))
    directVectGW = (W[0] - G[0], W[1] - G[1])
    D = (G[0] + bm.bust_span * directVectGW[0]/GW, G[1] + bm.bust_span * directVectGW[1]/GW)
    D2 = (D[0] + dartsB["dartIntake"] * directVectGW[0]/GW, D[1] + dartsB["dartIntake"] * directVectGW[1]/GW)
    D3 = (D2[0] + 1.25 * directVectGW[0]/GW, D2[1] + 1.25 * directVectGW[1]/GW)
    if dartsB["dartNum"] == 2:
     D4 = (D3[0] + dartsB["dartIntake"] * directVectGW[0]/GW, D3[1] + dartsB["dartIntake"] * directVectGW[1]/GW)
    else:
     D4 = (D3[0], D3[1])
    slopeGW = (W[1] - G[1])/(W[0] - G[0])
    invSlopeGW = -(1/slopeGW) # perpendicular to slopeGW, so negative reciprocal
    midDD2 = ((D[0] + D2[0])/2, (D[1] + D2[1])/2)
    xChangeMidDD2 = 5.5/math.sqrt(1 + pow(invSlopeGW, 2))
    yChangeMidDD2 = invSlopeGW * xChangeMidDD2
    DD2 = (midDD2[0] - xChangeMidDD2, midDD2[1] + yChangeMidDD2)
    midD3D4 = ((D3[0] + D4[0])/2, (D3[1] + D4[1])/2)
    xChangeMidD3D4 = 5.5/math.sqrt(1 + pow(invSlopeGW, 2))
    yChangeMidD3D4 = invSlopeGW * xChangeMidD3D4
    D3D4 = (midD3D4[0] - xChangeMidD3D4, midD3D4[1] + yChangeMidD3D4)
    EB2 = ((EB[0] + D4[0])/2, D4[1])
    IB3 = (I2[0], (EB[1] + I2[1])/2)
    return {"S":S, "Q":Q, "G":G, "D":D, "DD2":DD2, "D2":D2, "D3":D3, "D3D4":D3D4, "D4":D4, "EB2":EB2, "EB":EB, "IB3":IB3, "I2":I2, "I":I}

#endregion

# unisex
#region

def unisexSloperPantFront(bm):
    A = (0,0) # center waist
    B = (A[0], A[1] - bm.crotch_depth)
    C = (B[0], B[1] - bm.inseam)
    D = (A[0], A[1] - bm.knee_length)
    E = (A[0], A[1] - bm.hip_depth_front)
    B2 = (B[0] - bm.hip/8 + 0.5, B[1])
    E2 = (B2[0], E[1])
    E3 = (E2[0] + bm.hip/4 + 1, E2[1])
    B3 = (B2[0] - (-B2[0]/2), B2[1])
    A2 = (B2[0] + 0.5, A[1])
    A3 = (A2[0] + bm.waist/4 + 0.1, A[2])
    C2 = (C[0] - bm.ankle/4, C[1])
    C3 = (C[0] + bm.ankle/4, C[1])    
    D2 = (D[0] - bm.knee/4, D[1])
    D3 = (D[0] + bm.knee/4, D[1])

    B4 = (B2[0] - B2[0]/4, B2[1])
    E4 = (B4[0], E[1])
    F = (B4[0], -B4[1]/2)
    A4 = (B4[0] + 1, A[1])
    A5 = (A4[0], A4[1] + 0.5)
    B5 = (B2[0] + (B3[0] - B2[0])/2, B[1])
    B6 = (B5[0], B5[1] - 0.25)
    A6 = (A5[0] + math.sqrt(pow(bm.waist/4 + 0.5, 2) - pow(A5[1], 2)), A[1])
    E5 = (E2[0] + bm.waist/4 + 0.75, E[1])

    x = bm.croth_length - 1 # TODO: calculate pattern crotch lengths 

    distance_AD = calculate_distance(E2, E5)
    theta = calculate_rotation_angle(x, distance_AD)

    newE2 = rotate_point(E2, D, theta)
    newA5 = rotate_point(A5, D, theta)
    newA6 = rotate_point(A6, D, theta)



    # A = (0,0)
    # A1 = (A[0] - (bm.ankle/4 + 0.25), A[1])
    # A2 = (A[0] + (bm.ankle/4 + 0.25), A[1])
    # B = (A[0], A[1] + (bm.ankle_length - bm.knee_length))
    # B1 = (B[0] - (bm.knee/4 + 0.25), B[1])
    # B2 = (B[0] + (bm.knee/4 + 0.25), B[1])
    # C = (B[0], B[1] + (bm.knee_length - bm.crotch_depth))
    # C1 = (C[0] - (bm.thigh/4 + 0.25), C[1])
    # C2 = (C[0] + (bm.thigh/4 + 0.25), C[1])
    # D = (C[0], C[1] + (bm.crotch_depth - bm.hip_depth_front))
    # D1 = (D[0] - (bm.hip/4 + 0.25), D[1])
    # D2 = (D[0] + (bm.hip/4 + 0.25), D[1])
    # E = (D[0], D[1] + bm.ankle_length)
    # E1 = (E[0] - (bm.waist/4 + 0.25), E[1])
    # E2 = (E[0] + (bm.waist/4 + 0.25), E[1])

    # BC1 = (C1[0] - 0.5, (B[1] + C[1])/2)
    # C3 = (C1[0] - 0.5, C1[1] - 0.5)
    # slopeC1C3 = (C3[1] - C1[1])/(C3[0] - C1[0])
    # invSlopeC1C3 = -(1/slopeC1C3) # perpendicular to slopeC1C3, so negative reciprocal
    # slopeE1D1 = (D1[1] - E1[1])/(D1[0] - E1[0])

    # y = invSlopeC1C3(x) - (invSlopeC1C3(C1[0]) + C1[1])
    # y = slopeE1D1(x) - (slopeE1D1(D1[0]) + D1[1])
    # slopeE1D1 * X - (slopeE1D1(D1[0]) + D1[1]) = invSlopeC1C3 * X - (invSlopeC1C3(C1[0]) + C1[1])

    # xIntersectCDE = (-(slopeE1D1 * D1[0] + D1[1]) + (invSlopeC1C3 * C1[0] + C1[1])) / (invSlopeC1C3 - slopeE1D1)
    # yIntersectCDE = invSlopeC1C3 * xIntersectCDE - (invSlopeC1C3 * C1[0] + C1[1])

    # D3 = (xIntersectCDE, yIntersectCDE)

    # A = (0,0) # center front waist
    # B = (A[0], A[1] - bm.hip_depth_front)
    # C = (A[0], A[1] - bm.crotch_depth)
    # D = (A[0], A[1] - bm.knee_length)
    # E = (A[0], A[1] - bm.ankle_length)
    # F = (B[0] + bm.hip/4 + 0.25, B[1])
    # G = (A[0] + bm.waist/4 + 0.25, A[1])
    # H = (C[0] + bm.hip/10, C[1]) # crotch extension
    # I = (D[0] + bm.knee/2, D[1])
    # J = (E[0] + bm.ankle/2, E[1])

    Path = mpath.Path
    fig, ax = plt.subplots()
    # pp1 = mpatches.PathPatch(
    #      Path([],
    #           []),
    # fc="none", transform=ax.transData, linewidth=1)
   
    # ax.add_patch(pp1)
    
    points = {"A":A, "A1":A1, "A2":A2, "B":B, "B1":B1, "B2":B2, "C":C, "C1":C1, "C2":C2, "C3":C3, "D":D, "D1":D1, "D2":D2, "D3":D3,
              "E":E, "E1":E1, "E2":E2, "BC1":BC1}
    testingPoints(points)
    # Set Axis
    height = E[1]
    width = max(C2[0], D2[0], E2[0])
    setAxis(ax, fig, height, width)

    # Save PDF
    name = unisexSloperPantFront.__name__
    return savePDF(plt, name)

#endregion

# Helper functions
#region

def dartSkirt(waistHipDiff, front):
   if front:
      if waistHipDiff >= 11:
         return {"dartNum":2, "dartIntake":0.625}
      elif waistHipDiff >= 10:
         return {"dartNum":2, "dartIntake":0.5}
      elif 8 <= waistHipDiff < 10:
         return {"dartNum":2, "dartIntake":0.375}
      elif 3 <= waistHipDiff < 8:
         return {"dartNum":1, "dartIntake":0.5}
      else:
         return {"dartNum":0, "dartIntake":0} 
   else:
      if waistHipDiff >= 13:
         return {"dartNum":2, "dartIntake":1.375}
      elif waistHipDiff >= 12:
         return {"dartNum":2, "dartIntake":1.25}
      elif waistHipDiff >= 11:
         return {"dartNum":2, "dartIntake":1}
      elif waistHipDiff >= 10:
         return {"dartNum":2, "dartIntake":1.25}
      elif 8 <= waistHipDiff < 10:
         return {"dartNum":2, "dartIntake":1.25} 
      elif waistHipDiff >= 7:
         return {"dartNum":2, "dartIntake":0.875}
      elif waistHipDiff >= 7:
         return {"dartNum":2, "dartIntake":0.75}
      elif waistHipDiff >= 6:
         return {"dartNum":2, "dartIntake":0.675}
      elif waistHipDiff >= 5:
         return {"dartNum":1, "dartIntake":1}
      elif 3 <= waistHipDiff < 5:
         return {"dartNum":1, "dartIntake":0.75}
      else:
         return {"dartNum":0, "dartIntake":0}

def cubicBezierLen(P0, P1, P2, P3):
    # Convert the control points to NumPy arrays for vector operations
    P0, P1, P2, P3 = np.array(P0), np.array(P1), np.array(P2), np.array(P3)
    
    # Define the derivative function
    def derivative(t):
        return 3 * (1 - t)**2 * (P1 - P0) + 6 * (1 - t) * t * (P2 - P1) + 3 * t**2 * (P3 - P2)
    
    # Define a function to compute the magnitude of the derivative at a given t
    def derivMag(t):
        return np.linalg.norm(derivative(t))
    
    # Integrate the magnitude of the derivative over the interval [0, 1]
    length, _ = quad(derivMag, 0, 1)
    return length

def lowShoulderSquaredOff(topLowShoulder, lowShoulder, highShoulder):
   tlsls = topLowShoulder[1] - lowShoulder[1]
   hstls = topLowShoulder[0] - highShoulder[0]
   hsls = math.sqrt(pow(tlsls, 2) + pow(hstls, 2))
   ahstls = math.asin(hstls/hsls)
   acpsop = math.radians(90) - ahstls
   cpsop = 0.25 * math.sin(acpsop)
   lscp = math.sqrt(pow(0.25, 2) - pow(cpsop, 2))
   ControlPoint = (lowShoulder[0], lowShoulder[1] - lscp)
   SquaredOffPoint = (ControlPoint[0] - cpsop, ControlPoint[1])
   return {"ControlPoint":ControlPoint, "SquaredOffPoint":SquaredOffPoint}
   
def setAxis(ax, fig, height, width):
    # (0,0) starting point, 1 inch border
    edge = -1
    ax.axis([edge, width, edge, height])
    ax.axis('equal')
    ax.set_axis_off()
    fig.set_size_inches(width - (2 * edge), height - (2 * edge))

def savePDF(plt, name):
   plt.savefig("temp/" + name + ".pdf", bbox_inches = "tight", pad_inches = 0)
   return send_file('temp/' + name + '.pdf', as_attachment=True)

def testingPoints(points):
    for name, (x, y) in points.items():
        print(f'{name}, ({x}, {y})', flush=True)
        plt.scatter(x, y, label=name)
        annotation = f'{name}' #({x}, {y})'
        plt.annotate(annotation, xy=(x, y), xytext=(5, 5), textcoords='offset points')

# ChatGPT rotating points

# Step 1: Calculate the distance between point A and point D
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Step 2: Calculate the rotation angle theta (in radians)
def calculate_rotation_angle(x, distance_AD):
    return x / distance_AD

# Step 3: Rotate a point around a pivot (D)
def rotate_point(point, pivot, theta):
    # Applying the rotation formula
    x_new = pivot[0] + (point[0] - pivot[0]) * math.cos(theta) - (point[1] - pivot[1]) * math.sin(theta)
    y_new = pivot[1] + (point[0] - pivot[0]) * math.sin(theta) + (point[1] - pivot[1]) * math.cos(theta)
    return (x_new, y_new)

#endregion