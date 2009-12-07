/**
 * Autogenerated by Thrift
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 */
package nemo;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;
import java.util.Collections;
import java.util.BitSet;
import org.apache.log4j.Logger;

import org.apache.thrift.*;
import org.apache.thrift.meta_data.*;
import org.apache.thrift.protocol.*;

public class Synapse implements TBase, java.io.Serializable, Cloneable, Comparable<Synapse> {
  private static final TStruct STRUCT_DESC = new TStruct("Synapse");
  private static final TField TARGET_FIELD_DESC = new TField("target", TType.I32, (short)1);
  private static final TField DELAY_FIELD_DESC = new TField("delay", TType.I16, (short)2);
  private static final TField WEIGHT_FIELD_DESC = new TField("weight", TType.DOUBLE, (short)3);
  private static final TField PLASTIC_FIELD_DESC = new TField("plastic", TType.BOOL, (short)4);

  public int target;
  public static final int TARGET = 1;
  public short delay;
  public static final int DELAY = 2;
  public double weight;
  public static final int WEIGHT = 3;
  public boolean plastic;
  public static final int PLASTIC = 4;

  // isset id assignments
  private static final int __TARGET_ISSET_ID = 0;
  private static final int __DELAY_ISSET_ID = 1;
  private static final int __WEIGHT_ISSET_ID = 2;
  private static final int __PLASTIC_ISSET_ID = 3;
  private BitSet __isset_bit_vector = new BitSet(4);

  public static final Map<Integer, FieldMetaData> metaDataMap = Collections.unmodifiableMap(new HashMap<Integer, FieldMetaData>() {{
    put(TARGET, new FieldMetaData("target", TFieldRequirementType.DEFAULT, 
        new FieldValueMetaData(TType.I32)));
    put(DELAY, new FieldMetaData("delay", TFieldRequirementType.DEFAULT, 
        new FieldValueMetaData(TType.I16)));
    put(WEIGHT, new FieldMetaData("weight", TFieldRequirementType.DEFAULT, 
        new FieldValueMetaData(TType.DOUBLE)));
    put(PLASTIC, new FieldMetaData("plastic", TFieldRequirementType.DEFAULT, 
        new FieldValueMetaData(TType.BOOL)));
  }});

  static {
    FieldMetaData.addStructMetaDataMap(Synapse.class, metaDataMap);
  }

  public Synapse() {
    this.delay = (short)1;

    this.plastic = false;

  }

  public Synapse(
    int target,
    short delay,
    double weight,
    boolean plastic)
  {
    this();
    this.target = target;
    setTargetIsSet(true);
    this.delay = delay;
    setDelayIsSet(true);
    this.weight = weight;
    setWeightIsSet(true);
    this.plastic = plastic;
    setPlasticIsSet(true);
  }

  /**
   * Performs a deep copy on <i>other</i>.
   */
  public Synapse(Synapse other) {
    __isset_bit_vector.clear();
    __isset_bit_vector.or(other.__isset_bit_vector);
    this.target = other.target;
    this.delay = other.delay;
    this.weight = other.weight;
    this.plastic = other.plastic;
  }

  @Override
  public Synapse clone() {
    return new Synapse(this);
  }

  public int getTarget() {
    return this.target;
  }

  public Synapse setTarget(int target) {
    this.target = target;
    setTargetIsSet(true);
    return this;
  }

  public void unsetTarget() {
    __isset_bit_vector.clear(__TARGET_ISSET_ID);
  }

  // Returns true if field target is set (has been asigned a value) and false otherwise
  public boolean isSetTarget() {
    return __isset_bit_vector.get(__TARGET_ISSET_ID);
  }

  public void setTargetIsSet(boolean value) {
    __isset_bit_vector.set(__TARGET_ISSET_ID, value);
  }

  public short getDelay() {
    return this.delay;
  }

  public Synapse setDelay(short delay) {
    this.delay = delay;
    setDelayIsSet(true);
    return this;
  }

  public void unsetDelay() {
    __isset_bit_vector.clear(__DELAY_ISSET_ID);
  }

  // Returns true if field delay is set (has been asigned a value) and false otherwise
  public boolean isSetDelay() {
    return __isset_bit_vector.get(__DELAY_ISSET_ID);
  }

  public void setDelayIsSet(boolean value) {
    __isset_bit_vector.set(__DELAY_ISSET_ID, value);
  }

  public double getWeight() {
    return this.weight;
  }

  public Synapse setWeight(double weight) {
    this.weight = weight;
    setWeightIsSet(true);
    return this;
  }

  public void unsetWeight() {
    __isset_bit_vector.clear(__WEIGHT_ISSET_ID);
  }

  // Returns true if field weight is set (has been asigned a value) and false otherwise
  public boolean isSetWeight() {
    return __isset_bit_vector.get(__WEIGHT_ISSET_ID);
  }

  public void setWeightIsSet(boolean value) {
    __isset_bit_vector.set(__WEIGHT_ISSET_ID, value);
  }

  public boolean isPlastic() {
    return this.plastic;
  }

  public Synapse setPlastic(boolean plastic) {
    this.plastic = plastic;
    setPlasticIsSet(true);
    return this;
  }

  public void unsetPlastic() {
    __isset_bit_vector.clear(__PLASTIC_ISSET_ID);
  }

  // Returns true if field plastic is set (has been asigned a value) and false otherwise
  public boolean isSetPlastic() {
    return __isset_bit_vector.get(__PLASTIC_ISSET_ID);
  }

  public void setPlasticIsSet(boolean value) {
    __isset_bit_vector.set(__PLASTIC_ISSET_ID, value);
  }

  public void setFieldValue(int fieldID, Object value) {
    switch (fieldID) {
    case TARGET:
      if (value == null) {
        unsetTarget();
      } else {
        setTarget((Integer)value);
      }
      break;

    case DELAY:
      if (value == null) {
        unsetDelay();
      } else {
        setDelay((Short)value);
      }
      break;

    case WEIGHT:
      if (value == null) {
        unsetWeight();
      } else {
        setWeight((Double)value);
      }
      break;

    case PLASTIC:
      if (value == null) {
        unsetPlastic();
      } else {
        setPlastic((Boolean)value);
      }
      break;

    default:
      throw new IllegalArgumentException("Field " + fieldID + " doesn't exist!");
    }
  }

  public Object getFieldValue(int fieldID) {
    switch (fieldID) {
    case TARGET:
      return new Integer(getTarget());

    case DELAY:
      return new Short(getDelay());

    case WEIGHT:
      return new Double(getWeight());

    case PLASTIC:
      return new Boolean(isPlastic());

    default:
      throw new IllegalArgumentException("Field " + fieldID + " doesn't exist!");
    }
  }

  // Returns true if field corresponding to fieldID is set (has been asigned a value) and false otherwise
  public boolean isSet(int fieldID) {
    switch (fieldID) {
    case TARGET:
      return isSetTarget();
    case DELAY:
      return isSetDelay();
    case WEIGHT:
      return isSetWeight();
    case PLASTIC:
      return isSetPlastic();
    default:
      throw new IllegalArgumentException("Field " + fieldID + " doesn't exist!");
    }
  }

  @Override
  public boolean equals(Object that) {
    if (that == null)
      return false;
    if (that instanceof Synapse)
      return this.equals((Synapse)that);
    return false;
  }

  public boolean equals(Synapse that) {
    if (that == null)
      return false;

    boolean this_present_target = true;
    boolean that_present_target = true;
    if (this_present_target || that_present_target) {
      if (!(this_present_target && that_present_target))
        return false;
      if (this.target != that.target)
        return false;
    }

    boolean this_present_delay = true;
    boolean that_present_delay = true;
    if (this_present_delay || that_present_delay) {
      if (!(this_present_delay && that_present_delay))
        return false;
      if (this.delay != that.delay)
        return false;
    }

    boolean this_present_weight = true;
    boolean that_present_weight = true;
    if (this_present_weight || that_present_weight) {
      if (!(this_present_weight && that_present_weight))
        return false;
      if (this.weight != that.weight)
        return false;
    }

    boolean this_present_plastic = true;
    boolean that_present_plastic = true;
    if (this_present_plastic || that_present_plastic) {
      if (!(this_present_plastic && that_present_plastic))
        return false;
      if (this.plastic != that.plastic)
        return false;
    }

    return true;
  }

  @Override
  public int hashCode() {
    return 0;
  }

  public int compareTo(Synapse other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }

    int lastComparison = 0;
    Synapse typedOther = (Synapse)other;

    lastComparison = Boolean.valueOf(isSetTarget()).compareTo(isSetTarget());
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = TBaseHelper.compareTo(target, typedOther.target);
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = Boolean.valueOf(isSetDelay()).compareTo(isSetDelay());
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = TBaseHelper.compareTo(delay, typedOther.delay);
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = Boolean.valueOf(isSetWeight()).compareTo(isSetWeight());
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = TBaseHelper.compareTo(weight, typedOther.weight);
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = Boolean.valueOf(isSetPlastic()).compareTo(isSetPlastic());
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = TBaseHelper.compareTo(plastic, typedOther.plastic);
    if (lastComparison != 0) {
      return lastComparison;
    }
    return 0;
  }

  public void read(TProtocol iprot) throws TException {
    TField field;
    iprot.readStructBegin();
    while (true)
    {
      field = iprot.readFieldBegin();
      if (field.type == TType.STOP) { 
        break;
      }
      switch (field.id)
      {
        case TARGET:
          if (field.type == TType.I32) {
            this.target = iprot.readI32();
            setTargetIsSet(true);
          } else { 
            TProtocolUtil.skip(iprot, field.type);
          }
          break;
        case DELAY:
          if (field.type == TType.I16) {
            this.delay = iprot.readI16();
            setDelayIsSet(true);
          } else { 
            TProtocolUtil.skip(iprot, field.type);
          }
          break;
        case WEIGHT:
          if (field.type == TType.DOUBLE) {
            this.weight = iprot.readDouble();
            setWeightIsSet(true);
          } else { 
            TProtocolUtil.skip(iprot, field.type);
          }
          break;
        case PLASTIC:
          if (field.type == TType.BOOL) {
            this.plastic = iprot.readBool();
            setPlasticIsSet(true);
          } else { 
            TProtocolUtil.skip(iprot, field.type);
          }
          break;
        default:
          TProtocolUtil.skip(iprot, field.type);
          break;
      }
      iprot.readFieldEnd();
    }
    iprot.readStructEnd();


    // check for required fields of primitive type, which can't be checked in the validate method
    validate();
  }

  public void write(TProtocol oprot) throws TException {
    validate();

    oprot.writeStructBegin(STRUCT_DESC);
    oprot.writeFieldBegin(TARGET_FIELD_DESC);
    oprot.writeI32(this.target);
    oprot.writeFieldEnd();
    oprot.writeFieldBegin(DELAY_FIELD_DESC);
    oprot.writeI16(this.delay);
    oprot.writeFieldEnd();
    oprot.writeFieldBegin(WEIGHT_FIELD_DESC);
    oprot.writeDouble(this.weight);
    oprot.writeFieldEnd();
    oprot.writeFieldBegin(PLASTIC_FIELD_DESC);
    oprot.writeBool(this.plastic);
    oprot.writeFieldEnd();
    oprot.writeFieldStop();
    oprot.writeStructEnd();
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("Synapse(");
    boolean first = true;

    sb.append("target:");
    sb.append(this.target);
    first = false;
    if (!first) sb.append(", ");
    sb.append("delay:");
    sb.append(this.delay);
    first = false;
    if (!first) sb.append(", ");
    sb.append("weight:");
    sb.append(this.weight);
    first = false;
    if (!first) sb.append(", ");
    sb.append("plastic:");
    sb.append(this.plastic);
    first = false;
    sb.append(")");
    return sb.toString();
  }

  public void validate() throws TException {
    // check for required fields
    // check that fields of type enum have valid values
  }

}

