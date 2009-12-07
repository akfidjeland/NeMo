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

public class PipelineLength implements TBase, java.io.Serializable, Cloneable, Comparable<PipelineLength> {
  private static final TStruct STRUCT_DESC = new TStruct("PipelineLength");
  private static final TField INPUT_FIELD_DESC = new TField("input", TType.I32, (short)1);
  private static final TField OUTPUT_FIELD_DESC = new TField("output", TType.I32, (short)2);

  public int input;
  public static final int INPUT = 1;
  public int output;
  public static final int OUTPUT = 2;

  // isset id assignments
  private static final int __INPUT_ISSET_ID = 0;
  private static final int __OUTPUT_ISSET_ID = 1;
  private BitSet __isset_bit_vector = new BitSet(2);

  public static final Map<Integer, FieldMetaData> metaDataMap = Collections.unmodifiableMap(new HashMap<Integer, FieldMetaData>() {{
    put(INPUT, new FieldMetaData("input", TFieldRequirementType.DEFAULT, 
        new FieldValueMetaData(TType.I32)));
    put(OUTPUT, new FieldMetaData("output", TFieldRequirementType.DEFAULT, 
        new FieldValueMetaData(TType.I32)));
  }});

  static {
    FieldMetaData.addStructMetaDataMap(PipelineLength.class, metaDataMap);
  }

  public PipelineLength() {
  }

  public PipelineLength(
    int input,
    int output)
  {
    this();
    this.input = input;
    setInputIsSet(true);
    this.output = output;
    setOutputIsSet(true);
  }

  /**
   * Performs a deep copy on <i>other</i>.
   */
  public PipelineLength(PipelineLength other) {
    __isset_bit_vector.clear();
    __isset_bit_vector.or(other.__isset_bit_vector);
    this.input = other.input;
    this.output = other.output;
  }

  @Override
  public PipelineLength clone() {
    return new PipelineLength(this);
  }

  public int getInput() {
    return this.input;
  }

  public PipelineLength setInput(int input) {
    this.input = input;
    setInputIsSet(true);
    return this;
  }

  public void unsetInput() {
    __isset_bit_vector.clear(__INPUT_ISSET_ID);
  }

  // Returns true if field input is set (has been asigned a value) and false otherwise
  public boolean isSetInput() {
    return __isset_bit_vector.get(__INPUT_ISSET_ID);
  }

  public void setInputIsSet(boolean value) {
    __isset_bit_vector.set(__INPUT_ISSET_ID, value);
  }

  public int getOutput() {
    return this.output;
  }

  public PipelineLength setOutput(int output) {
    this.output = output;
    setOutputIsSet(true);
    return this;
  }

  public void unsetOutput() {
    __isset_bit_vector.clear(__OUTPUT_ISSET_ID);
  }

  // Returns true if field output is set (has been asigned a value) and false otherwise
  public boolean isSetOutput() {
    return __isset_bit_vector.get(__OUTPUT_ISSET_ID);
  }

  public void setOutputIsSet(boolean value) {
    __isset_bit_vector.set(__OUTPUT_ISSET_ID, value);
  }

  public void setFieldValue(int fieldID, Object value) {
    switch (fieldID) {
    case INPUT:
      if (value == null) {
        unsetInput();
      } else {
        setInput((Integer)value);
      }
      break;

    case OUTPUT:
      if (value == null) {
        unsetOutput();
      } else {
        setOutput((Integer)value);
      }
      break;

    default:
      throw new IllegalArgumentException("Field " + fieldID + " doesn't exist!");
    }
  }

  public Object getFieldValue(int fieldID) {
    switch (fieldID) {
    case INPUT:
      return new Integer(getInput());

    case OUTPUT:
      return new Integer(getOutput());

    default:
      throw new IllegalArgumentException("Field " + fieldID + " doesn't exist!");
    }
  }

  // Returns true if field corresponding to fieldID is set (has been asigned a value) and false otherwise
  public boolean isSet(int fieldID) {
    switch (fieldID) {
    case INPUT:
      return isSetInput();
    case OUTPUT:
      return isSetOutput();
    default:
      throw new IllegalArgumentException("Field " + fieldID + " doesn't exist!");
    }
  }

  @Override
  public boolean equals(Object that) {
    if (that == null)
      return false;
    if (that instanceof PipelineLength)
      return this.equals((PipelineLength)that);
    return false;
  }

  public boolean equals(PipelineLength that) {
    if (that == null)
      return false;

    boolean this_present_input = true;
    boolean that_present_input = true;
    if (this_present_input || that_present_input) {
      if (!(this_present_input && that_present_input))
        return false;
      if (this.input != that.input)
        return false;
    }

    boolean this_present_output = true;
    boolean that_present_output = true;
    if (this_present_output || that_present_output) {
      if (!(this_present_output && that_present_output))
        return false;
      if (this.output != that.output)
        return false;
    }

    return true;
  }

  @Override
  public int hashCode() {
    return 0;
  }

  public int compareTo(PipelineLength other) {
    if (!getClass().equals(other.getClass())) {
      return getClass().getName().compareTo(other.getClass().getName());
    }

    int lastComparison = 0;
    PipelineLength typedOther = (PipelineLength)other;

    lastComparison = Boolean.valueOf(isSetInput()).compareTo(isSetInput());
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = TBaseHelper.compareTo(input, typedOther.input);
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = Boolean.valueOf(isSetOutput()).compareTo(isSetOutput());
    if (lastComparison != 0) {
      return lastComparison;
    }
    lastComparison = TBaseHelper.compareTo(output, typedOther.output);
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
        case INPUT:
          if (field.type == TType.I32) {
            this.input = iprot.readI32();
            setInputIsSet(true);
          } else { 
            TProtocolUtil.skip(iprot, field.type);
          }
          break;
        case OUTPUT:
          if (field.type == TType.I32) {
            this.output = iprot.readI32();
            setOutputIsSet(true);
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
    oprot.writeFieldBegin(INPUT_FIELD_DESC);
    oprot.writeI32(this.input);
    oprot.writeFieldEnd();
    oprot.writeFieldBegin(OUTPUT_FIELD_DESC);
    oprot.writeI32(this.output);
    oprot.writeFieldEnd();
    oprot.writeFieldStop();
    oprot.writeStructEnd();
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("PipelineLength(");
    boolean first = true;

    sb.append("input:");
    sb.append(this.input);
    first = false;
    if (!first) sb.append(", ");
    sb.append("output:");
    sb.append(this.output);
    first = false;
    sb.append(")");
    return sb.toString();
  }

  public void validate() throws TException {
    // check for required fields
    // check that fields of type enum have valid values
  }

}

